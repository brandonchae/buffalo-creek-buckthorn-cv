"""
fetch_live_naip_and_inaturalist.py
Automated fetcher for:
1. Official USGS/USDA NAIP 4-Band (RGB + NIR) GeoTIFF over Buffalo Creek Forest Preserve (Lake County, IL)
2. Research-grade iNaturalist GPS observations of Rhamnus cathartica in Lake County, IL
"""
import urllib.request
import json
import pandas as pd

BBOX_WGS84 = (-88.0120, 42.1590, -88.0010, 42.1660)

def download_usgs_naip_geotiff(out_tif="buffalo_creek_naip_4band_live.tif"):
    minx, miny, maxx, maxy = BBOX_WGS84
    url = (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
        f"?bbox={minx},{miny},{maxx},{maxy}&bboxSR=4326&size=306,256&imageSR=26916"
        "&format=tiff&pixelType=U8&noDataInterpretation=esriNoDataMatchAny"
        "&interpolation=+RSP_BilinearInterpolation&f=image"
    )
    print("Downloading 4-Band NAIP GeoTIFF (EPSG:26916) from USGS National Map...")
    urllib.request.urlretrieve(url, out_tif)
    print(f"Saved live NAIP GeoTIFF to {out_tif}")

def fetch_inaturalist_buckthorn(out_csv="inaturalist_lake_county_buckthorn.csv"):
    url = (
        "https://api.inaturalist.org/v1/observations"
        "?taxon_name=Rhamnus%20cathartica&quality_grade=research"
        "&swlat=42.10&swlng=-88.20&nelat=42.35&nelng=-87.85&per_page=200"
    )
    print("Querying iNaturalist API v1 for research-grade Rhamnus cathartica in Lake County, IL...")
    req = urllib.request.Request(url, headers={"User-Agent": "BuffaloCreekBuckthornMapper/2.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = []
    for obs in data.get("results", []):
        geojson = obs.get("geojson")
        if geojson and "coordinates" in geojson:
            lon, lat = geojson["coordinates"]
            rows.append({
                "INAT_ID": obs["id"],
                "OBSERVED_ON": obs.get("observed_on"),
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "POSITIONAL_ACCURACY_M": obs.get("positional_accuracy"),
                "SPECIES": "Rhamnus cathartica",
                "BUCKTHORN_PRESENT": 1
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} research-grade iNaturalist observations to {out_csv}")

if __name__ == "__main__":
    download_usgs_naip_geotiff()
    fetch_inaturalist_buckthorn()
