import os
import time
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio

load_dotenv()

app = FastAPI(title="Aushadhi Terminology Service")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI missing in environment")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
DB_NAME = os.getenv("DB_NAME", "aushadhi")

CS_URL = os.getenv("NAMASTE_CS_URL", "urn:namaste:ayurveda")
CS_VERSION = os.getenv("NAMASTE_CS_VERSION", "2025-09")
VS_URL = os.getenv("NAMASTE_VS_URL", "urn:namaste:ayurveda:all")

CM_TM2_CUR_URL = os.getenv("CM_TM2_CUR_URL", "urn:map:namaste-to-tm2:curated")
CM_MMS_SUG_URL = os.getenv("CM_MMS_SUG_URL", "urn:map:namaste-to-mms:suggestions")
CM_MMS_CUR_URL = os.getenv("CM_MMS_CUR_URL", "urn:map:namaste-to-mms:curated")

ICD_FHIR_BASE = os.getenv("ICD_FHIR_BASE", "https://id.who.int/fhir")
ICD_TM2_CS_URL = os.getenv("ICD11_TM2_CS_URL", "http://id.who.int/icd/release/11/tm2")
ICD_TM2_VERSION = os.getenv("ICD11_TM2_VERSION", "2025-01")
ICD_MMS_CS_URL = os.getenv("ICD11_MMS_CS_URL", "http://id.who.int/icd/release/11/mms")
ICD_MMS_VERSION = os.getenv("ICD11_MMS_VERSION", "2025-01")

ICD_CLIENT_ID = os.getenv("ICD_CLIENT_ID", "")
ICD_CLIENT_SECRET = os.getenv("ICD_CLIENT_SECRET", "")
ICD_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
ICD_SCOPE = "icdapi_access"

_icd_token: Optional[str] = None
_icd_exp: float = 0.0

async def _get_icd_token() -> str:
    global _icd_token, _icd_exp
    now = time.time()
    if _icd_token and now < _icd_exp - 60:
        return _icd_token
    if not ICD_CLIENT_ID or not ICD_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="ICD API credentials missing (ICD_CLIENT_ID/ICD_CLIENT_SECRET)")
    data = {
        "grant_type": "client_credentials",
        "scope": ICD_SCOPE,
        "client_id": ICD_CLIENT_ID,
        "client_secret": ICD_CLIENT_SECRET,
    }
    async with httpx.AsyncClient(timeout=15) as cx:
        r = await cx.post(ICD_TOKEN_URL, data=data, headers={"Accept": "application/json"})
        r.raise_for_status()
        tok = r.json()
        _icd_token = tok["access_token"]
        _icd_exp = now + int(tok.get("expires_in", 3600))
        return _icd_token

async def _icd_headers() -> Dict[str, str]:
    t = await _get_icd_token()
    return {
        "Authorization": f"Bearer {t}",
        "API-Version": "v2",
        "Accept": "application/fhir+json",
    }

def coll(db, name):
    return db[name]

async def ensure_indexes(db):
    await coll(db, "codesystems").create_index([("url", 1), ("version", 1)], unique=True)
    await coll(db, "valuesets").create_index([("url", 1), ("version", 1)], unique=True)
    await coll(db, "conceptmaps").create_index([("url", 1), ("version", 1)], unique=True)
    await coll(db, "proposals").create_index([("sourceCode", 1), ("icdCode", 1), ("version", 1)], unique=True)

def build_artifacts_from_df(df: pd.DataFrame):
    concepts = []
    tm2_curated_elements = []
    mms_suggestion_elements = []
    pattern = r'^\s*([^\s].*?)\s*\(\s*([^)]+)\s*\)\s*$'
    for _, r in df.fillna("").iterrows():
        raw_code = str(r.get("NAMC_CODE", "")).strip()
        if not raw_code:
            continue
        display = (str(r.get("Name English", "")).strip() or str(r.get("NAMC_term", "")).strip()) or None
        long_def = str(r.get("Long_definition", "")).strip()
        short_def = str(r.get("Short_definition", "")).strip()
        definition = (long_def or short_def) or None
        m = pd.Series([raw_code]).str.replace("\u00A0", " ", regex=False).str.extract(pattern, expand=True)
        has_tm2_pair = not m.isna().values.any()
        if has_tm2_pair:
            tm2_code = str(m.iat[0, 0]).strip()
            namaste_code = str(m.iat[0, 1]).strip()
        else:
            tm2_code = None
            namaste_code = raw_code

        designations = []
        if r.get("NAMC_term"): designations.append({"language": "und", "value": str(r["NAMC_term"])})
        if r.get("NAMC_term_diacritical"): designations.append({"language": "und", "value": str(r["NAMC_term_diacritical"])})
        if r.get("NAMC_term_DEVANAGARI"): designations.append({"language": "und", "value": str(r["NAMC_term_DEVANAGARI"])})

        properties = []
        if r.get("Ontology_branches"): properties.append({"code": "ontology", "valueString": str(r["Ontology_branches"])})
        if r.get("Name English Under Index"): properties.append({"code": "indexName", "valueString": str(r["Name English Under Index"])})
        if r.get("Primary Index Related"): properties.append({"code": "indexRelated", "valueString": str(r["Primary Index Related"])})
        if tm2_code: properties.append({"code": "tm2Code", "valueString": tm2_code})

        concept = {"code": namaste_code}
        if display: concept["display"] = display
        if definition: concept["definition"] = definition
        if designations: concept["designation"] = designations
        if properties: concept["property"] = properties
        concepts.append(concept)

        if tm2_code:
            tm2_curated_elements.append({
                "code": namaste_code,
                "display": display,
                "target": [{
                    "code": tm2_code,
                    "display": display,
                    "equivalence": "equivalent",
                    "comment": f"Auto-mapped from raw NAMC_CODE '{raw_code}'"
                }]
            })
        else:
            mms_suggestion_elements.append({"code": namaste_code, "display": display})

    code_system = {
        "resourceType": "CodeSystem",
        "url": CS_URL,
        "version": CS_VERSION,
        "name": "NAMASTE_Ayurveda",
        "status": "active",
        "content": "complete",
        "property": [
            {"code": "ontology", "description": "Ontology branches", "type": "string"},
            {"code": "indexName", "description": "English under index", "type": "string"},
            {"code": "indexRelated", "description": "Primary index related", "type": "string"},
            {"code": "tm2Code", "description": "Mapped TM2 code (if present in source)", "type": "string"}
        ],
        "concept": concepts
    }
    value_set = {
        "resourceType": "ValueSet",
        "url": VS_URL,
        "version": CS_VERSION,
        "name": "NAMASTE_Ayurveda_All",
        "status": "active",
        "compose": {"include": [{"system": CS_URL, "version": CS_VERSION}]}
    }
    cm_tm2_curated = {
        "resourceType": "ConceptMap",
        "url": CM_TM2_CUR_URL,
        "version": CS_VERSION,
        "name": "NamasteToTM2_Curated",
        "status": "active",
        "group": [{
            "source": CS_URL,
            "sourceVersion": CS_VERSION,
            "target": ICD_TM2_CS_URL,
            "targetVersion": ICD_TM2_VERSION,
            "element": tm2_curated_elements,
            "unmapped": {"mode": "other-map", "url": CM_MMS_SUG_URL}
        }]
    }
    cm_mms_suggestions = {
        "resourceType": "ConceptMap",
        "url": CM_MMS_SUG_URL,
        "version": CS_VERSION,
        "name": "NamasteToMMS_Suggestions",
        "status": "draft",
        "group": [{
            "source": CS_URL,
            "sourceVersion": CS_VERSION,
            "target": ICD_MMS_CS_URL,
            "targetVersion": ICD_MMS_VERSION,
            "element": mms_suggestion_elements
        }]
    }
    return code_system, value_set, cm_tm2_curated, cm_mms_suggestions

def merge_code_system_additive(existing, incoming):
    if not existing: return incoming
    have = {c.get("code") for c in (existing.get("concept") or [])}
    additions = [c for c in (incoming.get("concept") or []) if c.get("code") not in have]
    concept = (existing.get("concept") or []) + additions
    props = incoming.get("property") or existing.get("property")
    return {**existing, **incoming, "property": props, "concept": concept}

def map_by_source(elements):
    return {e.get("code"): e for e in (elements or [])}

def merge_tm2_curated_skip_existing(existing_cur, incoming_cur):
    cur_existing = ((existing_cur or {}).get("group") or [{}])[0].get("element") or []
    by_src = map_by_source(cur_existing)
    for e in ((incoming_cur.get("group") or [{}])[0].get("element") or []):
        if e.get("code") not in by_src:
            by_src[e.get("code")] = e
    mergedCur = {**(existing_cur or incoming_cur), **incoming_cur}
    group0 = {**((incoming_cur.get("group") or [{}])[0])}
    group0["element"] = list(by_src.values())
    mergedCur["group"] = [group0]
    return mergedCur

def merge_mms_suggestions(existing_sug, incoming_sug, tm2_cur, mms_cur):
    sug_existing = ((existing_sug or {}).get("group") or [{}])[0].get("element") or []
    sug_map = map_by_source(sug_existing)
    tm2_src = set(map_by_source(((tm2_cur or {}).get("group") or [{}])[0].get("element") or []).keys())
    mms_src = set(map_by_source(((mms_cur or {}).get("group") or [{}])[0].get("element") or []).keys())
    for src in tm2_src.union(mms_src):
        sug_map.pop(src, None)
    for e in ((incoming_sug.get("group") or [{}])[0].get("element") or []):
        code = e.get("code")
        if code not in tm2_src and code not in mms_src and code not in sug_map:
            sug_map[code] = e
    mergedSug = {**(existing_sug or incoming_sug), **incoming_sug}
    group0 = {**((incoming_sug.get("group") or [{}])[0])}
    group0["element"] = list(sug_map.values())
    mergedSug["group"] = [group0]
    return mergedSug

async def upsert_merged(db, coll_name, filter_doc, resource):
    now = datetime.utcnow().isoformat()
    res = dict(resource)
    res.pop("_id", None)
    res.pop("_createdAt", None)
    await coll(db, coll_name).update_one(
        filter_doc,
        {"$set": {**res, "_lastUpdated": now}, "$setOnInsert": {"_createdAt": now}},
        upsert=True
    )

async def validate_icd_mms(code: str) -> Dict[str, Any]:
    params = {"url": ICD_MMS_CS_URL, "version": ICD_MMS_VERSION, "code": code}
    headers = await _icd_headers()
    async with httpx.AsyncClient(timeout=20) as cx:
        r = await cx.get(f"{ICD_FHIR_BASE}/CodeSystem/$validate-code", params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
    ok = next((x.get("valueBoolean") for x in data.get("parameter", []) if x.get("name") == "result"), False)
    display = next((x.get("valueString") for x in data.get("parameter", []) if x.get("name") == "display"), "")
    message = ""
    for p in data.get("parameter", []):
        if p.get("name") == "message":
            if "valueString" in p:
                message = p["valueString"]
            elif "resource" in p:
                message = str(p["resource"])
    return {"ok": bool(ok), "display": display or "", "message": message}

async def lookup_icd_mms(code: str) -> Dict[str, Any]:
    params = {"system": ICD_MMS_CS_URL, "version": ICD_MMS_VERSION, "code": code}
    headers = await _icd_headers()
    async with httpx.AsyncClient(timeout=20) as cx:
        r = await cx.get(f"{ICD_FHIR_BASE}/CodeSystem/$lookup", params=params, headers=headers)
        r.raise_for_status()
        return r.json()

@app.get("/config")
async def config():
    return {
        "cs_url": CS_URL,
        "cs_version": CS_VERSION,
        "tm2_canonical": ICD_TM2_CS_URL,
        "tm2_release": ICD_TM2_VERSION,
        "mms_canonical": ICD_MMS_CS_URL,
        "mms_release": ICD_MMS_VERSION,
        "icd_fhir_base": ICD_FHIR_BASE
    }

@app.post("/ingest/namaste-csv")
async def ingest(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False, engine="python")
        code_system_new, value_set, cm_tm2_cur_new, cm_mms_sug_new = build_artifacts_from_df(df)
        db = client[DB_NAME]
        await ensure_indexes(db)

        csFilter = {"url": code_system_new["url"], "version": code_system_new["version"]}
        csExisting = await coll(db, "codesystems").find_one(csFilter)
        csMerged = merge_code_system_additive(csExisting, code_system_new)
        await upsert_merged(db, "codesystems", csFilter, csMerged)

        await upsert_merged(db, "valuesets", {"url": value_set["url"], "version": value_set["version"]}, value_set)

        tm2Filter = {"url": CM_TM2_CUR_URL, "version": CS_VERSION}
        tm2Existing = await coll(db, "conceptmaps").find_one(tm2Filter)
        tm2Merged = merge_tm2_curated_skip_existing(tm2Existing, cm_tm2_cur_new)
        await upsert_merged(db, "conceptmaps", tm2Filter, tm2Merged)

        mmsCurFilter = {"url": CM_MMS_CUR_URL, "version": CS_VERSION}
        mmsCurExisting = await coll(db, "conceptmaps").find_one(mmsCurFilter)
        if not mmsCurExisting:
            mmsCurExisting = {
                "resourceType": "ConceptMap",
                "url": CM_MMS_CUR_URL,
                "version": CS_VERSION,
                "name": "NamasteToMMS_Curated",
                "status": "active",
                "group": [{
                    "source": CS_URL,
                    "sourceVersion": CS_VERSION,
                    "target": ICD_MMS_CS_URL,
                    "targetVersion": ICD_MMS_VERSION,
                    "element": []
                }]
            }
            await upsert_merged(db, "conceptmaps", mmsCurFilter, mmsCurExisting)

        mmsSugFilter = {"url": CM_MMS_SUG_URL, "version": CS_VERSION}
        mmsSugExisting = await coll(db, "conceptmaps").find_one(mmsSugFilter)
        mmsSugMerged = merge_mms_suggestions(mmsSugExisting, cm_mms_sug_new, tm2Merged, mmsCurExisting)
        await upsert_merged(db, "conceptmaps", mmsSugFilter, mmsSugMerged)

        return {
            "ok": True,
            "db": DB_NAME,
            "stats": {
                "cs_concepts_before": (len(csExisting.get("concept")) if csExisting and csExisting.get("concept") else 0),
                "cs_concepts_after": len(csMerged.get("concept") or []),
                "tm2_curated_after": len(tm2Merged["group"][0]["element"]),
                "mms_suggestions_after": len(mmsSugMerged["group"][0]["element"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions")
async def list_suggestions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    db = client[DB_NAME]
    sugFilter = {"url": CM_MMS_SUG_URL, "version": CS_VERSION}
    cm = await coll(db, "conceptmaps").find_one(sugFilter)
    elements = ((cm or {}).get("group") or [{}])[0].get("element") or []
    items = elements[offset:offset + limit]
    items_trim = [{"code": e.get("code"), "display": e.get("display", "")} for e in items]
    return {"total": len(elements), "items": items_trim}

@app.get("/suggestions/by-code")
async def get_suggestions_for_code(sourceCode: str = Query(..., description="NAMASTE code")):
    db = client[DB_NAME]
    sugFilter = {"url": CM_MMS_SUG_URL, "version": CS_VERSION}
    cm = await coll(db, "conceptmaps").find_one(sugFilter)
    elements = ((cm or {}).get("group") or [{}])[0].get("element") or []
    el = next((e for e in elements if e.get("code") == sourceCode), None)
    targets = (el.get("target") if el else []) or []
    targets_sorted = sorted(targets, key=lambda t: float(t.get("score", 0.0)), reverse=True)
    out = [{"code": t.get("code"), "display": t.get("display", ""), "comment": t.get("comment", ""), "score": float(t.get("score", 0.0))} for t in targets_sorted]
    return {"sourceCode": sourceCode, "count": len(out), "targets": out}

@app.get("/concepts/by-code")
async def get_concept_by_code(code: str = Query(...)):
    db = client[DB_NAME]
    cs = await coll(db, "codesystems").find_one({"url": CS_URL, "version": CS_VERSION})
    concepts = (cs or {}).get("concept") or []
    c = next((x for x in concepts if x.get("code") == code), None)
    if not c:
        return {"ok": False, "concept": None}
    props = {p.get("code"): p.get("valueString") for p in (c.get("property") or []) if isinstance(p, dict)}
    designations = [d.get("value") for d in (c.get("designation") or []) if isinstance(d, dict) and d.get("value")]
    concept = {
        "code": c.get("code"),
        "display": c.get("display", ""),
        "definition": c.get("definition", ""),
        "summary": "",
        "keywords": " | ".join([p for k, p in props.items() if k in ("indexName", "indexRelated")]) if props else "",
        "designations": designations
    }
    return {"ok": True, "concept": concept}

@app.get("/curated/by-code")
async def get_curated_for_code(sourceCode: str = Query(...)):
    db = client[DB_NAME]
    curFilter = {"url": CM_MMS_CUR_URL, "version": CS_VERSION}
    cm = await coll(db, "conceptmaps").find_one(curFilter)
    elements = ((cm or {}).get("group") or [{}])[0].get("element") or []
    el = next((e for e in elements if e.get("code") == sourceCode), None)
    targets = (el.get("target") if el else []) or []
    out = []
    for t in targets:
        out.append({
            "code": t.get("code"),
            "display": t.get("display") or "",
            "equivalence": t.get("equivalence") or "",
            "comment": t.get("comment") or ""
        })
    return {"sourceCode": sourceCode, "count": len(out), "targets": out}

@app.get("/proposals/by-code")
async def get_proposals_for_code(sourceCode: str = Query(...)):
    db = client[DB_NAME]
    cur = coll(db, "proposals").find({"sourceCode": sourceCode, "version": CS_VERSION})
    items: List[Dict[str, Any]] = []
    async for d in cur:
        d.pop("_id", None)
        items.append(d)
    items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    return {"sourceCode": sourceCode, "items": items}

@app.get("/proposals")
async def list_proposals(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sourceCode: Optional[str] = Query(None)
):
    db = client[DB_NAME]
    q: Dict[str, Any] = {"version": CS_VERSION}
    if sourceCode:
        q["sourceCode"] = sourceCode
    cur = coll(db, "proposals").find(q)
    all_items: List[Dict[str, Any]] = []
    async for d in cur:
        d.pop("_id", None)
        all_items.append(d)
    all_items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    items = all_items[offset:offset + limit]
    return {"total": len(all_items), "items": items}

@app.get("/icd/validate")
async def api_validate_icd(code: str = Query(..., description="ICD-11 MMS code candidate")):
    try:
        result = await validate_icd_mms(code)
        return {"icd": code, "canonical": ICD_MMS_CS_URL, "release": ICD_MMS_VERSION, **result}
    except httpx.HTTPStatusError as e:
        detail = f"{e.response.status_code} {e.response.text}"
        raise HTTPException(status_code=502, detail=f"WHO FHIR validate error: {detail}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"WHO FHIR validate error: {e}")

@app.get("/icd/lookup")
async def api_lookup_icd(code: str = Query(..., description="ICD-11 MMS code")):
    try:
        data = await lookup_icd_mms(code)
        return {"icd": code, "canonical": ICD_MMS_CS_URL, "release": ICD_MMS_VERSION, "lookup": data}
    except httpx.HTTPStatusError as e:
        detail = f"{e.response.status_code} {e.response.text}"
        raise HTTPException(status_code=502, detail=f"WHO FHIR lookup error: {detail}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"WHO FHIR lookup error: {e}")

@app.post("/mapping/propose")
async def propose_mapping(
    sourceCode: str = Body(...),
    icdCode: str = Body(...),
    score: float = Body(..., ge=0.0, le=1.0),
    suggestedEquivalence: str = Body(..., description="equivalent|narrower|wider|inexact"),
    note: Optional[str] = Body(None),
    display: Optional[str] = Body(None),
    comment: Optional[str] = Body(None)
):
    db = client[DB_NAME]
    await ensure_indexes(db)
    doc = {
        "sourceCode": sourceCode,
        "icdCode": icdCode,
        "score": score,
        "suggestedEquivalence": suggestedEquivalence,
        "note": note or "",
        "version": CS_VERSION,
        "icdCanonical": ICD_MMS_CS_URL,
        "icdRelease": ICD_MMS_VERSION,
        "status": "proposed",
        "display": display or "",
        "comment": comment or "",
        "createdAt": datetime.utcnow().isoformat()
    }
    await coll(db, "proposals").update_one(
        {"sourceCode": sourceCode, "icdCode": icdCode, "version": CS_VERSION},
        {"$set": doc},
        upsert=True
    )
    return {"ok": True, "proposal": doc}

@app.post("/suggestions/propose")
async def propose_from_suggestion(
    sourceCode: str = Body(...),
    icdCode: str = Body(...),
    suggestedEquivalence: str = Body(..., description="equivalent|narrower|wider|inexact"),
    note: Optional[str] = Body(None)
):
    db = client[DB_NAME]
    sugFilter = {"url": CM_MMS_SUG_URL, "version": CS_VERSION}
    cm = await coll(db, "conceptmaps").find_one(sugFilter)
    elements = ((cm or {}).get("group") or [{}])[0].get("element") or []
    el = next((e for e in elements if e.get("code") == sourceCode), None)
    if not el:
        raise HTTPException(status_code=404, detail=f"No suggestions found for {sourceCode}")
    target = next((t for t in (el.get("target") or []) if str(t.get("code")) == icdCode), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"ICD candidate {icdCode} not found in suggestions for {sourceCode}")
    score = float(target.get("score", 0.0))
    display = target.get("display") or ""
    comment = target.get("comment") or ""
    return await propose_mapping(
        sourceCode=sourceCode,
        icdCode=icdCode,
        score=score,
        suggestedEquivalence=suggestedEquivalence,
        note=note,
        display=display,
        comment=comment
    )

@app.post("/mapping/promote")
async def promote_mapping(
    sourceCode: str = Body(...),
    icdCode: str = Body(...),
    equivalence: str = Body(..., description="equivalent|narrower|wider|inexact|unmatched"),
    revalidate: bool = Body(True),
    reviewer: Optional[str] = Body(None),
    extraComment: Optional[str] = Body(None)
):
    db = client[DB_NAME]
    await ensure_indexes(db)

    icd_display = ""
    if revalidate and equivalence != "unmatched":
        try:
            v = await validate_icd_mms(icdCode)
            if not v["ok"]:
                return {"ok": False, "error": f"ICD MMS candidate invalid for release {ICD_MMS_VERSION}: {v['message']}"}
            icd_display = v.get("display") or ""
        except httpx.HTTPStatusError as e:
            detail = f"{e.response.status_code} {e.response.text}"
            raise HTTPException(status_code=502, detail=f"WHO FHIR validate error: {detail}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"WHO FHIR validate error: {e}")

    # Source display from CodeSystem
    cs = await coll(client[DB_NAME], "codesystems").find_one({"url": CS_URL, "version": CS_VERSION})
    concepts = (cs or {}).get("concept") or []
    src_concept = next((x for x in concepts if x.get("code") == sourceCode), None)
    namaste_display = (src_concept or {}).get("display", "")

    mmsCurFilter = {"url": CM_MMS_CUR_URL, "version": CS_VERSION}
    mmsSugFilter = {"url": CM_MMS_SUG_URL, "version": CS_VERSION}
    mmsCurExisting = await coll(db, "conceptmaps").find_one(mmsCurFilter)
    mmsSugExisting = await coll(db, "conceptmaps").find_one(mmsSugFilter)

    if not mmsCurExisting:
        mmsCurExisting = {
            "resourceType": "ConceptMap",
            "url": CM_MMS_CUR_URL,
            "version": CS_VERSION,
            "name": "NamasteToMMS_Curated",
            "status": "active",
            "group": [{
                "source": CS_URL,
                "sourceVersion": CS_VERSION,
                "target": ICD_MMS_CS_URL,
                "targetVersion": ICD_MMS_VERSION,
                "element": []
            }]
        }

    ts = datetime.utcnow().isoformat()
    cm_comment = f"WHO {ICD_MMS_VERSION}; eq={equivalence}; reviewer={reviewer or 'unknown'}; {extraComment or ''}".strip()

    # Append-or-merge target into curated element
    cur_elements = ((mmsCurExisting or {}).get("group") or [{}])[0].get("element") or []
    by_src = {e.get("code"): e for e in cur_elements}
    if sourceCode in by_src:
        existing_el = by_src[sourceCode]
        existing_el["display"] = existing_el.get("display") or namaste_display
        tgt_list = existing_el.get("target") or []
        found = False
        for t in tgt_list:
            if str(t.get("code")) == icdCode:
                t["equivalence"] = equivalence
                t["display"] = icd_display
                t["comment"] = cm_comment
                found = True
                break
        if not found:
            tgt_list.append({
                "code": icdCode,
                "display": icd_display,
                "equivalence": equivalence,
                "comment": cm_comment
            })
        existing_el["target"] = tgt_list
    else:
        by_src[sourceCode] = {
            "code": sourceCode,
            "display": namaste_display,
            "target": [{
                "code": icdCode,
                "display": icd_display,
                "equivalence": equivalence,
                "comment": cm_comment
            }]
        }
    mmsCurExisting.setdefault("group", [{}])
    if not mmsCurExisting["group"]:
        mmsCurExisting["group"] = [{}]
    mmsCurExisting["group"][0]["target"] = ICD_MMS_CS_URL
    mmsCurExisting["group"][0]["targetVersion"] = ICD_MMS_VERSION
    mmsCurExisting["group"][0]["element"] = list(by_src.values())

    # Remove only the promoted target from suggestions
    if mmsSugExisting:
        sug_elements = ((mmsSugExisting or {}).get("group") or [{}])[0].get("element") or []
        for e in sug_elements:
            if e.get("code") == sourceCode:
                tgt = e.get("target") or []
                tgt_after = [t for t in tgt if str(t.get("code")) != icdCode]
                e["target"] = tgt_after
                break
        mmsSugExisting["group"][0]["element"] = sug_elements

    await upsert_merged(db, "conceptmaps", mmsCurFilter, mmsCurExisting)
    if mmsSugExisting:
        await upsert_merged(db, "conceptmaps", mmsSugFilter, mmsSugExisting)

    # Mark only this proposal as promoted
    await coll(db, "proposals").update_one(
        {"sourceCode": sourceCode, "icdCode": icdCode, "version": CS_VERSION},
        {"$set": {"status": "promoted", "promotedAt": ts}}
    )

    return {
        "ok": True,
        "curated_count": len(mmsCurExisting["group"][0]["element"])
    }

@app.post("/mapping/reject")
async def reject_mapping(
    sourceCode: str = Body(...),
    icdCode: Optional[str] = Body(None),
    reviewer: Optional[str] = Body(None),
    reason: Optional[str] = Body(None)
):
    db = client[DB_NAME]
    await ensure_indexes(db)
    q = {"sourceCode": sourceCode, "version": CS_VERSION}
    if icdCode:
        q["icdCode"] = icdCode
    res = await coll(db, "proposals").delete_many(q)
    return {
        "ok": True,
        "deleted_proposals": res.deleted_count,
        "kept_in_suggestions": True,
        "reviewer": reviewer or "",
        "reason": reason or ""
    }