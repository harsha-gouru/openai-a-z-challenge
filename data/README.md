# Data Directory Guide

This folder acts as the single **data volume** for the Amazon Deep Insights project.  
Large geospatial and text assets are **never** committed to Git; instead they are
downloaded on-demand or recovered through DVC and cached here.

| Sub-folder | Purpose | Typical Contents |
|------------|---------|------------------|
| **raw/** | Immutable, original downloads exactly as received from the external source. | • LiDAR `.laz` / `.las` tiles<br>• Remote-sensing rasters (`.tif`) <br>• Shapefiles / GeoJSON site catalogues<br>• PDFs & HTML scraped papers |
| **processed/** | Cleaned and derived artefacts ready for analysis. These are _deterministic_ outputs of our pipelines. | • Digital Elevation Models (DEMs)<br>• Canopy Height Models (CHMs)<br>• Clipped & re-projected rasters<br>• Feature tables (`.parquet`, `.csv`) |
| **vector_db/** | On-disk persistence layer for the ChromaDB vector store used by the RAG system. | • `chroma.sqlite3` <br>• Embedding parquet/shard files |
| **tmp/** (git-ignored) | Scratch space for intermediate, non-deterministic files. Cleared automatically. | • Pipeline temp files<br>• Download resumptions |
| **README.md** | *You are here* — overview and best practices. | |

## Best Practices

1. **DVC First** – add any large or regenerable file with  
   `dvc add data/raw/<file>` then commit the `.dvc` metafile.
2. **Immutable Raw** – never edit files in `raw/`; create a processed version
   instead.
3. **Consistency** – all rasters must be re-projected to _EPSG:4326_ before
   placement in `processed/`.
4. **Disk Hygiene** – periodic `dvc gc` keeps cache size manageable.

Feel free to add subdirectories if a new data modality emerges, but update this
README so the team stays aligned.
