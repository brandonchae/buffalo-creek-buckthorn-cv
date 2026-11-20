# Buffalo Creek Invasive Buckthorn Geospatial Mapper

Multispectral remote-sensing and spatial machine learning pipeline to detect and map **European Buckthorn (*Rhamnus cathartica*)** canopy encroachment across **Buffalo Creek Forest Preserve** (Lake County, IL) using 4-band (RGB + Near-Infrared) NAIP aerial imagery and field-verified GPS ground-truth waypoints.

## Ecological & Remote-Sensing Basis
European buckthorn exhibits **extended autumn leaf retention**, remaining photosynthetically active in late October and November after native Midwestern oak-hickory woodlands and tallgrass prairies have senesced. By combining late-autumn **NDVI**, **Green-Red Vegetation Index (GRVI)**, and **$5 \times 5$ spatial canopy texture**, this classifier isolates dense invasive buckthorn thickets for targeted Lake County Forest Preserve District (LCFPD) restoration crews.

## Model Performance (5-Fold Stratified Cross-Validation)
- **Cross-Validated Accuracy:** `97.5%`
- **Cross-Validated ROC-AUC:** `1.000`
- **Estimated Priority Removal Zone:** `6.01 acres` (`15.2%` of surveyed 39.5-acre parcel)

## How to Run
```bash
pip install numpy pandas matplotlib scipy scikit-learn
python buckthorn_mapper.py
```
Running the script automatically generates `buffalo_creek_gps_ground_truth.csv` and `buckthorn_invasion_heatmap.png`.