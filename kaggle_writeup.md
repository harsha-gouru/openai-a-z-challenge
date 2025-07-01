# Amazon Geoglyph Detection

## Abstract
We developed a lightweight pipeline for detecting linear features in Amazon LiDAR tiles.

## Method
Using NumPy and SciPy, we computed gradient-based edges and extracted connected components as candidate lines. RasterIO loads tiles and Folium visualises results.

## Results
Running the pipeline on two sample tiles produced one candidate line per tile. Both were tagged as `POSSIBLE_NEW` since no reference overlap exists.

## Next Steps
Incorporate GPT-4 vision scoring when API access is available and expand training data for better accuracy.
