"""Fetch candidate iNaturalist records; never creates training labels.

NAIP acquisition is intentionally explicit: download the appropriate authoritative
USDA/USGS product for the chosen footprint/year, inspect its metadata, then record
its provenance in data/imagery_provenance.json. This script will not guess imagery.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://api.inaturalist.org/v1/observations"
TAXON = "Rhamnus cathartica"

def fetch_candidates(out_csv: Path, bbox: tuple[float,float,float,float], max_pages: int = 5):
    """Save candidate records with IDs, dates, grade, taxon, and uncertainty.

    iNaturalist records are leads for review, not ground-truth presence labels.
    """
    import pandas as pd
    west,south,east,north=bbox
    rows=[]
    for page in range(1,max_pages+1):
        params={"taxon_name":TAXON,"quality_grade":"research","swlat":south,"swlng":west,
                "nelat":north,"nelng":east,"per_page":200,"page":page,"order":"asc","order_by":"observed_on"}
        req=Request(API+"?"+urlencode(params),headers={"User-Agent":"BuffaloCreekBuckthornResearch/1.0 (data provenance workflow)"})
        with urlopen(req,timeout=60) as response: payload=json.load(response)
        results=payload.get("results",[])
        for obs in results:
            coords=(obs.get("geojson") or {}).get("coordinates")
            if not coords: continue
            rows.append({"observation_id":str(obs.get("id")),"longitude":coords[0],"latitude":coords[1],
                "observation_date":obs.get("observed_on"),"taxon_name":(obs.get("taxon") or {}).get("name"),
                "quality_grade":obs.get("quality_grade"),"positional_accuracy_m":obs.get("positional_accuracy"),
                "source_url":f"https://www.inaturalist.org/observations/{obs.get('id')}",
                "verification_status":"candidate_unverified"})
        if len(results)<200 or len(rows)>=payload.get("total_results",0): break
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    out_csv.with_suffix(".provenance.json").write_text(json.dumps({"source":API,"taxon":TAXON,"bbox_wgs84":bbox,
        "retrieved_utc":datetime.now(timezone.utc).isoformat(),"records":len(rows),
        "interpretation":"Candidate observations only; not validated ground truth. Absence of records is not absence of species."},indent=2))
    return len(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox",nargs=4,type=float,required=True,metavar=("WEST","SOUTH","EAST","NORTH"),help="WGS84 bounding box")
    ap.add_argument("--output",type=Path,default=Path("data/inaturalist_candidates.csv"))
    ap.add_argument("--pages",type=int,default=5)
    args=ap.parse_args()
    n=fetch_candidates(args.output,tuple(args.bbox),args.pages)
    print(f"Saved {n} candidate records to {args.output}; these are not ground-truth labels.")
    print("NAIP is not fetched automatically. Acquire and document a real USDA/USGS image product separately.")

if __name__=="__main__": main()
