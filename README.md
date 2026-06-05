# Aushadhi: Ayurvedic Terminology Mapping Service (Smart India Hackathon)

An automated terminology mapping platform designed to bridge the gap between traditional Ayurvedic medicine (**NAMASTE**) and global standards (**WHO ICD-11 MMS**).

## The Challenge
During **SIH 2025**, the goal was to convert raw Ayurvedic term definitions into interoperable digital health formats. Traditional terms often lack direct 1:1 mappings in modern medical systems, requiring a semantic-heavy approach for accurate cross-referencing.

## Tech Stack
* **Backend:** FastAPI, Motor (Async MongoDB), HL7 FHIR Standards.
* **AI/NLP:** HuggingFace, Sentence-Transformers (SapBERT, MPNet Ensemble).
* **Data Science:** Pandas, NumPy, Cosine Similarity scoring.
* **Industry Standards:** ICD-11 MMS, FHIR CodeSystem/ConceptMap.

## Architecture & Workflow


1.  **Data Ingestion:** Raw CSVs containing NAMASTE terms are parsed and transformed into **FHIR CodeSystems** and **ValueSets**.
2.  **Semantic Mapping Engine:**
    * Generates text embeddings for term definitions using an **Ensemble Model (SapBERT + MPNet)**.
    * Queries the **WHO ICD-11 API** to fetch candidate destination entities.
    * Calculates **Cosine Similarity** between embeddings to rank the top 5 suggestions.
3.  **Human-in-the-loop Validation:** A **Propose-and-Promote** workflow allows reviewers to validate and promote AI-generated mappings to curated status.
4.  **Interoperability:** Final mappings are served as **FHIR ConceptMaps**, enabling integration with modern Electronic Health Records (EHRs).

## Key Components
* `ingest_ayu.py`: Handles the FHIR resource creation and MongoDB upserts.
* `semantic_mapper.py` (NLP logic): Implements the semantic search and embedding fusion.
* `main.py`: Provides the API endpoints for translation, lookups, and the proposal lifecycle.

## Getting Started
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure `.env` with your **ICD_CLIENT_ID** and **MONGODB_URI**.
4. Run the API: `uvicorn backend.main:app --reload`.
