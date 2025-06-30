# Project Plan – OpenAI A-Z Challenge  
**Working Title:** “Amazon Deep Insights – Unearthing Forest Secrets with LiDAR & LLMs”

---

## 1. Project Overview
The Amazon rainforest conceals immense ecological complexity and thousands of undocumented archaeological features.  
Our hackathon goal is to build a Retrieval-Augmented-Generation (RAG) prototype that:

* Ingests LiDAR, remote-sensing, and open research datasets
* Detects & classifies forest structure and possible anthropogenic patterns
* Allows scientists to query findings conversationally through an OpenAI LLM

The result will accelerate discovery of hidden archaeological sites and support conservation decisions.

---

## 2. Objectives
| # | Objective | Success Metric |
|---|-----------|----------------|
| O1 | Assemble a unified, searchable knowledge base of Amazon LiDAR & ancillary datasets | ≥ 5 key datasets downloaded, parsed, embedded |
| O2 | Identify potential archaeological earthworks & tall-tree hotspots from LiDAR | ≥ 80 % precision on small validation set |
| O3 | Expose insights via chat interface powered by RAG | Demo answers ≥ 10 domain questions with relevant citations |
| O4 | Publish reproducible code & documentation | Public repo, Dockerfile, quick-start guide |

---

## 3. Key Data Sources
(links already curated in `Links.md`)

* ORNL DAAC LiDAR Forest Inventory (ds_id = 1644, 1515, 1514, 1302)  
* OpenTopography OT.042013.4326.1 high-resolution point clouds  
* Kaggle “Forested Areas – Pará/Brazil 1514” & “Amazonas/Brazil” shapefiles  
* Zenodo repositories on Amazon biomass & LiDAR tiles (e.g., 7689909)  
* Published research articles (Science 2022, Nature 2021), CAA Journal 2018  
* Wikipedia extract of known archaeological sites (Bornstein notebook)  
* WeatherAPI & other environmental layers (rainfall, elevation) for context

---

## 4. Methodology

1. **Data Acquisition & Storage**  
   • Download LiDAR LAS/LAZ tiles → store in S3 / local volume  
   • Store tabular / raster data in PostgreSQL + PostGIS  
   • Convert articles & PDFs to text chunks for RAG

2. **Pre-processing**  
   • LiDAR: ground/non-ground classification -> DEM & Canopy Height Model  
   • Generate vegetation metrics (max height, canopy density, roughness)  
   • Align ancillary rasters to common CRS (EPSG:4326)

3. **Feature Extraction & Analytics**  
   • Use PDAL & ForestTools to derive micro-topography & tree crowns  
   • Train lightweight classifier (e.g., XGBoost) to flag candidate mounds, geoglyphs, giant trees  
   • Validate with known site coordinates from Wikipedia dataset

4. **Embedding & Vector Store**  
   • Text: OpenAI text-embedding-3-small → ChromaDB  
   • Spatial features: convert geojson attribute summaries to text for embedding

5. **RAG Pipeline**  
   • User question → embed → retrieve top-k chunks → augment prompt  
   • LLM (gpt-4o mini) produces answer + references

6. **Visualization**  
   • Quickleaflet / Kepler.gl map for bounding-box preview  
   • plas.io link outs for 3-D point-cloud inspection

---

## 5. Implementation Steps

| Phase | Tasks | Tools |
|-------|-------|-------|
| Setup | Repo scaffold, Docker, .env for keys | GitHub, Docker |
| Data Ingestion | Write fetch scripts per dataset, log with DVC | Python, requests, pdal |
| Pre-processing | PDAL pipelines, GDAL rasterization | PDAL, GDAL |
| Analytics | Feature engineering notebooks, model training | scikit-learn, xgboost |
| RAG Backend | Build FastAPI service: embed, store, retrieve | FastAPI, ChromaDB |
| Frontend | Streamlit chat UI + map | Streamlit, Mapbox |
| Testing/Docs | Unit tests, README, demo notebook | pytest, mkdocs |
| Demo Day | Deploy on HuggingFace Spaces | HF Spaces |

---

## 6. Timeline (48-hour Hackathon)

| Hour | Milestone |
|------|-----------|
| 0-2  | Kick-off, repo init, task assignment |
| 2-8  | Download & stage priority datasets |
| 8-14 | LiDAR preprocessing pipeline functional |
| 14-20| Feature extraction + baseline classifier |
| 20-26| Text corpus scraping & embedding |
| 26-32| Vector store + RAG endpoint |
| 32-38| Streamlit UI with chat & map |
| 38-42| Evaluation, prompt tuning |
| 42-46| Polish docs, record demo video |
| 46-48| Submission & buffer |

---

## 7. Team Roles

* Data Engineer – Ingestion & storage pipelines  
* Geo-Analyst – LiDAR processing & feature extraction  
* ML Engineer – Classifier & embeddings  
* Backend Dev – RAG API, vector DB  
* Frontend Dev – UI & visualization  
* PM/Scribe – Timelines, docs, submission

---

## 8. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Large LiDAR file sizes | Slow download/processing | Clip to AOIs, process small tiles first |
| Limited ground truth for archaeology | Low model accuracy | Use unsupervised anomaly detection, manual review |
| LLM hallucinations | Misleading answers | Always show citations & retrieval score |

---

## 9. Deliverables

1. Public GitHub repo with MIT license  
2. Hosted demo (HuggingFace Spaces)  
3. 2-minute video walkthrough  
4. PDF poster summarizing results

---

**Let’s build it!**
