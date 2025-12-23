import os
import sys
import json
import time
import argparse
import logging
import re
import html
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict

import numpy as np
import requests
from pymongo import MongoClient
from pymongo.client_session import ClientSession
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("mms-semantic")

def enable_http_debug():
    from http.client import HTTPConnection
    HTTPConnection.debuglevel = 1
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").propagate = True
    logging.getLogger("requests.packages.urllib3").propagate = True
    log.debug("Enabled HTTP wire debug")

def enable_pymongo_debug():
    logging.getLogger("pymongo").setLevel(logging.DEBUG)
    logging.getLogger("pymongo.command").setLevel(logging.DEBUG)
    logging.getLogger("pymongo.connection").setLevel(logging.DEBUG)
    logging.getLogger("pymongo.serverSelection").setLevel(logging.DEBUG)
    log.debug("Enabled PyMongo debug")

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    log.error("MONGODB_URI missing in environment")
    sys.exit(2)

ICD_MMS_RELEASE = os.getenv("ICD_MMS_RELEASE", "2025-01")
ICD_API_BASE = f"https://id.who.int/icd/release/11/{ICD_MMS_RELEASE}/mms"

ICD_CLIENT_ID = os.getenv("ICD_CLIENT_ID", "")
ICD_CLIENT_SECRET = os.getenv("ICD_CLIENT_SECRET", "")
ICD_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
ICD_SCOPE = "icdapi_access"

DB_NAME = "aushadhi"
CODESYSTEMS_COL = "codesystems"
CONCEPTMAPS_COL = "conceptmaps"
ICD_CACHE_COL = "icd_mms_cache"

SUGGESTION_CM_URL = "urn:map:namaste-to-mms:suggestions"
GROUP_SOURCE = "urn:namaste:ayurveda"
GROUP_TARGET = "http://id.who.int/icd/release/11/mms"

MAX_SUGGESTIONS = 5

SAPBERT_MODEL = os.getenv("SAPBERT_MODEL", "UMCU/SapBERT-from-PubMedBERT-fulltext_bf16")
MPNET_MODEL = os.getenv("MPNET_MODEL", "sentence-transformers/all-mpnet-base-v2")
W_SAP = float(os.getenv("SAPBERT_WEIGHT", "0.5"))
W_MP = float(os.getenv("MPNET_WEIGHT", "0.5"))
W_SUM = max(W_SAP + W_MP, 1e-9)

model_sap = SentenceTransformer(SAPBERT_MODEL)
model_mp = SentenceTransformer(MPNET_MODEL)
log.info("Loaded ensemble models: SapBERT=%s | MPNet=%s | weights=(%.2f, %.2f)", SAPBERT_MODEL, MPNET_MODEL, W_SAP, W_MP)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
codesystems_col = db[CODESYSTEMS_COL]
conceptmaps_col = db[CONCEPTMAPS_COL]
icd_cache_col = db[ICD_CACHE_COL]
log.info("Connected to MongoDB and collections initialized")

STOPWORDS = set("""
a an the and or of to for with on in by from as at into about over under between during within through
is are was were be being been have has had do does did this that these those such
""".split())

def normalize_str(x: Optional[str]) -> str:
    return (x or "").strip()

def strip_html(s: str) -> str:
    if not s:
        return s
    clean = re.sub(r"<[^>]+>", "", s)
    return html.unescape(clean)

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10
    return v / n

def ensemble_encode(texts: List[str]) -> np.ndarray:
    emb_sap = model_sap.encode(texts, normalize_embeddings=True)
    emb_mp = model_mp.encode(texts, normalize_embeddings=True)
    fused = (W_SAP * emb_sap + W_MP * emb_mp) / W_SUM
    fused = _l2norm(fused)
    return fused

def generate_embedding(text: str) -> List[float]:
    vec = ensemble_encode([text])[0]
    return vec.tolist()

def cosine_similarity(vec1, vec2) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(np.dot(v1, v2) / denom)

def _extract_property_value(prop: Dict) -> Optional[str]:
    for key in ["valueString", "valueCode", "valueBoolean", "valueInteger", "valueDecimal", "valueCoding", "valueDateTime"]:
        if key in prop and prop[key] is not None:
            val = prop[key]
            return str(val) if not isinstance(val, str) else val
    return None

def get_summary_keywords_from_codesystem(code: str) -> Tuple[str, str, str]:
    cs_doc = codesystems_col.find_one({"url": GROUP_SOURCE})
    if not cs_doc:
        log.warning("CodeSystem %s not found", GROUP_SOURCE)
        return "", "", ""
    concepts = cs_doc.get("concept") or cs_doc.get("concepts") or []
    target = next((c for c in concepts if c.get("code") == code), None)
    if not target:
        log.debug("Concept %s not found in CodeSystem %s", code, GROUP_SOURCE)
        return "", "", ""
    display = normalize_str(target.get("display"))
    summary = ""
    keywords = ""
    for prop in target.get("property", []):
        pcode = prop.get("code")
        if pcode == "Summary":
            summary = normalize_str(_extract_property_value(prop) or "")
        elif pcode == "Keywords":
            keywords = normalize_str(_extract_property_value(prop) or "")
    return summary, keywords, display

_icd_token: Optional[str] = None
_icd_exp: float = 0.0

def get_icd_token() -> str:
    global _icd_token, _icd_exp
    now = time.time()
    if _icd_token and now < _icd_exp - 60:
        return _icd_token
    if not ICD_CLIENT_ID or not ICD_CLIENT_SECRET:
        log.error("ICD API credentials missing (ICD_CLIENT_ID/ICD_CLIENT_SECRET)")
        raise RuntimeError("ICD API credentials missing")
    data = {"grant_type": "client_credentials", "scope": ICD_SCOPE, "client_id": ICD_CLIENT_ID, "client_secret": ICD_CLIENT_SECRET}
    r = requests.post(ICD_TOKEN_URL, data=data, headers={"Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    tok = r.json()
    _icd_token = tok["access_token"]
    _icd_exp = now + int(tok.get("expires_in", 3600))
    log.info("ICD token acquired; expires in %ss", int(tok.get("expires_in", 3600)))
    return _icd_token

def icd_headers() -> Dict[str, str]:
    t = get_icd_token()
    return {"Authorization": f"Bearer {t}", "API-Version": "v2", "Accept": "application/json", "Accept-Language": "en"}

def build_query_text(summary: str, keywords: str, fallback: str) -> str:
    base = (keywords or summary or fallback or "").strip()
    tokens = re.findall(r"[A-Za-z0-9\-]+", base.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens[:16])

def split_keywords(keywords: str, fallback: str) -> List[str]:
    if not keywords:
        return [fallback] if fallback else []
    parts = re.split(r"[;,|]\s*|\s{2,}", keywords)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def fetch_icd_entity_detail_by_id(entity_id: str) -> Optional[Dict]:
    url = f"{ICD_API_BASE}/{entity_id}"
    try:
        r = requests.get(url, headers=icd_headers(), timeout=20)
        log.debug("ICD entity GET %s status=%s", url, r.status_code)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

def title_string_from_detail(j: Dict) -> str:
    t = j.get("title")
    if isinstance(t, dict):
        return t.get("@value") or t.get("value") or ""
    if isinstance(t, str):
        return t
    return ""

def extract_definition_from_entity(j: Dict) -> Optional[str]:
    for k in ("definition", "longDefinition", "description", "content", "narrative"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            val = v.get("en") or v.get("@value") or v.get("value")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None

def search_icd_destination_entities(q: str) -> List[Dict]:
    url = f"{ICD_API_BASE}/search"
    headers = icd_headers()
    try:
        if len(q) <= 256:
            params = {"q": q, "flatResults": "true"}
            log.info("ICD search GET: q_len=%d", len(q))
            r = requests.get(url, params=params, headers=headers, timeout=30)
        else:
            log.info("ICD search POST: q_len=%d", len(q))
            form = {"q": q, "flatResults": "true"}
            r = requests.post(url, data=form, headers=headers, timeout=30)
        log.debug("ICD search %s %s status=%s", r.request.method, r.url, r.status_code)
        if r.status_code != 200:
            log.warning("ICD search non-200: %s body=%s", r.status_code, r.text[:500])
            return []
        j = r.json()
        ents = j.get("destinationEntities") or []
        out: List[Dict] = []
        for it in ents:
            ent_id = it.get("id") or it.get("@id") or it.get("entityId") or ""
            code = it.get("code") or it.get("theCode") or ""
            title = ""
            t = it.get("title")
            if isinstance(t, dict):
                title = t.get("@value") or t.get("value") or ""
            elif isinstance(t, str):
                title = t
            if ent_id:
                if isinstance(ent_id, str) and ent_id.startswith("http"):
                    ent_id = ent_id.rstrip("/").split("/")[-1]
                out.append({"entity_id": ent_id, "code": code, "display": strip_html(title)})
        log.info("ICD destination entities: %d", len(out))
        return out
    except requests.RequestException as e:
        log.exception("ICD search error: %s", e)
        return []

def ensure_icd_cache_entry(code: str, text_for_embed: str, apply_changes: bool) -> Optional[List[float]]:
    if apply_changes:
        doc = icd_cache_col.find_one({"code": code})
        if doc and doc.get("embedding"):
            log.debug("Cache hit for ICD %s (has embedding)", code)
            return doc["embedding"]
        emb = generate_embedding(text_for_embed) if text_for_embed else None
        if doc:
            icd_cache_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"title": text_for_embed[:200], "embedding": emb, "embedding_model": f"ensemble:{SAPBERT_MODEL}+{MPNET_MODEL}"}}
            )
        else:
            icd_cache_col.insert_one({
                "code": code,
                "title": text_for_embed[:200],
                "embedding": emb,
                "embedding_model": f"ensemble:{SAPBERT_MODEL}+{MPNET_MODEL}"
            })
        return emb
    else:
        return generate_embedding(text_for_embed) if text_for_embed else None

def build_candidates_from_cache(namaste_vec: List[float]) -> List[Dict]:
    candidates: List[Dict] = []
    for icd_doc in icd_cache_col.find({}):
        icd_embedding = icd_doc.get("embedding")
        if not icd_embedding:
            continue
        score = cosine_similarity(namaste_vec, icd_embedding)
        candidates.append({"code": icd_doc["code"], "display": icd_doc.get("title", ""), "score": score})
    log.debug("Cache scan produced %d candidate scores", len(candidates))
    return candidates

def find_group_index(cm_doc: Dict) -> int:
    groups = cm_doc.get("group", []) or []
    for i, g in enumerate(groups):
        if g.get("source") == GROUP_SOURCE and g.get("target") == GROUP_TARGET:
            return i
    return 0 if groups else -1

def upsert_targets_in_conceptmap(cm_doc: Dict, group_idx: int, elem_idx: int, targets: List[Dict], session: Optional[ClientSession] = None):
    update_path = f"group.{group_idx}.element.{elem_idx}.target"
    conceptmaps_col.update_one(
        {"_id": cm_doc["_id"]},
        {"$set": {update_path: targets, "_lastUpdated": datetime.now(timezone.utc).isoformat()}},
        session=session
    )
    log.debug("Wrote %d targets at %s", len(targets), update_path)

def semantic_search_unmapped(apply_changes: bool = False, preview_file: Optional[str] = None):
    cm_doc = conceptmaps_col.find_one({"url": SUGGESTION_CM_URL})
    if not cm_doc:
        log.error("ConceptMap not found: %s", SUGGESTION_CM_URL)
        return

    group_idx = find_group_index(cm_doc)
    if group_idx < 0:
        log.error("No matching group in ConceptMap for source=%s target=%s", GROUP_SOURCE, GROUP_TARGET)
        return

    group = cm_doc.get("group", [])[group_idx]
    elements = group.get("element", []) or []
    log.info("Loaded suggestions: %d elements", len(elements))

    proposed_updates = []

    for i, el in enumerate(elements):
        code = normalize_str(el.get("code"))
        if not code:
            continue

        targets_existing = el.get("target", [])
        if isinstance(targets_existing, list) and len(targets_existing) > 0:
            log.debug("Skip %s (already has %d targets)", code, len(targets_existing))
            continue

        summary, keywords_raw, cs_display = get_summary_keywords_from_codesystem(code)
        el_display = normalize_str(el.get("display"))
        namaste_text = (f"{summary} {keywords_raw}".strip() or el_display or cs_display or code)
        if not namaste_text:
            log.debug("Skip %s (no text to embed)", code)
            continue

        namaste_vec = generate_embedding(namaste_text)

        kw_list = split_keywords(keywords_raw, build_query_text(summary, keywords_raw, (el_display or cs_display or code)))
        seen_codes: set = set()
        candidates: List[Dict] = []

        for kw in kw_list:
            ents = search_icd_destination_entities(kw)
            if not ents:
                continue
            for ent in ents[:5]:
                ent_id = ent.get("entity_id")
                ent_code = ent.get("code") or ""
                ent_disp = strip_html(ent.get("display") or "")
                dedup_key = ent_code or ent_id
                if not dedup_key or dedup_key in seen_codes:
                    continue
                detail_json = fetch_icd_entity_detail_by_id(ent_id)
                if not detail_json:
                    continue
                definition = extract_definition_from_entity(detail_json)
                if not definition:
                    continue
                cand_text = definition
                icd_emb = ensure_icd_cache_entry(dedup_key, cand_text, apply_changes=apply_changes)
                score = cosine_similarity(namaste_vec, icd_emb) if icd_emb is not None else 0.5
                if not ent_disp:
                    ent_disp = strip_html(title_string_from_detail(detail_json))
                comment = definition
                candidates.append({"code": ent_code or ent_id, "display": ent_disp, "score": score, "comment": comment})
                seen_codes.add(dedup_key)

        if len(candidates) < MAX_SUGGESTIONS:
            compact_q = build_query_text(summary, keywords_raw, (el_display or cs_display or code))
            url = f"{ICD_API_BASE}/search"
            headers = icd_headers()
            try:
                if len(compact_q) <= 256:
                    r = requests.get(url, params={"q": compact_q, "flatResults": "true"}, headers=headers, timeout=30)
                else:
                    r = requests.post(url, data={"q": compact_q, "flatResults": "true"}, headers=headers, timeout=30)
                if r.status_code == 200:
                    j = r.json()
                    raw = j.get("results") or []
                    for it in raw:
                        code2 = it.get("code") or it.get("theCode") or ""
                        disp2 = it.get("title") or it.get("name") or it.get("label") or ""
                        if not code2 or code2 in seen_codes:
                            continue
                        detail_json = fetch_icd_entity_detail_by_id(code2)
                        definition = extract_definition_from_entity(detail_json or {}) if detail_json else None
                        cand_text = (definition or disp2 or code2)
                        icd_emb = ensure_icd_cache_entry(code2, cand_text, apply_changes=apply_changes)
                        score = cosine_similarity(namaste_vec, icd_emb) if icd_emb is not None else 0.5
                        clean_disp = strip_html(disp2)
                        comment = definition or ""
                        candidates.append({"code": code2, "display": clean_disp, "score": score, "comment": comment})
                        seen_codes.add(code2)
                        if len(candidates) >= MAX_SUGGESTIONS:
                            break
                else:
                    log.warning("Fallback search non-200: %s", r.status_code)
            except requests.RequestException as e:
                log.debug("Fallback search error: %s", e)

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:MAX_SUGGESTIONS]

        targets = []
        for c in candidates:
            t = {"code": c["code"], "score": round(float(c["score"]), 4)}
            if c.get("display"):
                t["display"] = c["display"]
            if c.get("comment"):
                t["comment"] = f"def: {c['comment'][:360]}..." if len(c["comment"]) > 360 else f"def: {c['comment']}"
            targets.append(t)

        log.info("Proposed %d candidates for %s (top_score=%.3f)", len(targets), code, candidates[0]["score"] if candidates else 0.0)

        proposed_updates.append({
            "element_index": i,
            "code": code,
            "display": el.get("display", ""),
            "text_used": namaste_text,
            "targets": targets,
        })

    if not apply_changes:
        log.info("[PREVIEW] Would update %d elements in ConceptMap %s (group idx %d)", len(proposed_updates), SUGGESTION_CM_URL, group_idx)
        for p in proposed_updates[:10]:
            log.info("- Code %s (element %d): %d targets", p["code"], p["element_index"], len(p["targets"]))
            for t in p["targets"]:
                log.info("  • %s | %s | score=%.3f", t.get("code",""), t.get("display",""), t.get("score", 0.0))
        if len(proposed_updates) > 10:
            log.info("... and %d more", len(proposed_updates) - 10)
        if preview_file:
            payload = {"conceptMapUrl": SUGGESTION_CM_URL, "groupIndex": group_idx, "timestamp": datetime.now(timezone.utc).isoformat(), "proposals": proposed_updates}
            with open(preview_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log.info("[PREVIEW] Wrote proposals to %s", preview_file)
        return

    with client.start_session() as session:
        def txn_ops(sess: ClientSession):
            for p in proposed_updates:
                update_path = f"group.{group_idx}.element.{p['element_index']}.target"
                conceptmaps_col.update_one(
                    {"_id": cm_doc["_id"]},
                    {"$set": {update_path: p["targets"], "_lastUpdated": datetime.now(timezone.utc).isoformat()}},
                    session=sess
                )
        session.with_transaction(txn_ops)

    log.info("[APPLY] Updated %d elements in ConceptMap %s", len(proposed_updates), SUGGESTION_CM_URL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suggest MMS targets with SapBERT+MPNet ensemble; all keywords; 5 entities/keyword; top-5 overall.")
    parser.add_argument("--apply", action="store_true", help="Apply changes to the ConceptMap in a MongoDB transaction.")
    parser.add_argument("--preview-file", type=str, default=None, help="Write proposed updates to a JSON file (preview mode only).")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logs.")
    parser.add_argument("--http-debug", action="store_true", help="Enable HTTP wire debug logs.")
    parser.add_argument("--mongo-debug", action="store_true", help="Enable PyMongo debug logs.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Root logger set to DEBUG")
    if args.http_debug:
        enable_http_debug()
    if args.mongo_debug:
        enable_pymongo_debug()

    semantic_search_unmapped(apply_changes=args.apply, preview_file=args.preview_file)