"""Reproducible buckthorn classifier. Requires real imagery and verified labels.

No synthetic or fallback data are generated. See README for input schema and caveats.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix)

LOG = logging.getLogger("buckthorn")
SEED = 2026
TARGET_CRS = "EPSG:26916"

def safe_index(a, b, offset=0.0):
    den = a + b + offset
    return np.divide(a-b, den, out=np.full_like(a, np.nan, dtype=float), where=np.abs(den)>1e-8)

def raster_features(arr):
    """Band-first uint/float NAIP with [R,G,B,NIR]; indices and 5/7 px texture."""
    if arr.ndim != 3 or arr.shape[0] < 4:
        raise ValueError("NAIP raster must have at least four bands in R,G,B,NIR order")
    r,g,b,nir = [arr[i].astype(float) for i in range(4)]
    ndvi=safe_index(nir,r); gndvi=safe_index(nir,g); grvi=safe_index(g,r); savi=1.5*safe_index(nir,r,0.5)
    def local_std(x,size):
        valid=np.isfinite(x); x0=np.nan_to_num(x)
        m=uniform_filter(x0,size=size); m2=uniform_filter(x0*x0,size=size)
        count=uniform_filter(valid.astype(float),size=size)
        return np.sqrt(np.maximum(np.divide(m2,count,out=np.full_like(m,np.nan),where=count>0)-np.square(np.divide(m,count,out=np.full_like(m,np.nan),where=count>0)),0))
    return np.stack([r,g,b,nir,ndvi,gndvi,grvi,savi,
        uniform_filter(np.nan_to_num(ndvi),size=5),local_std(ndvi,5),local_std(nir,7)])

FEATURES=["red","green","blue","nir","ndvi","gndvi","grvi","savi","ndvi_mean_5","ndvi_sd_5","nir_sd_7"]

def validate_inputs(tif, obs):
    if not tif.is_file() or not obs.is_file(): raise FileNotFoundError(f"Required inputs missing: {tif} and/or {obs}")
    with rasterio.open(tif) as ds:
        if ds.crs is None: raise ValueError("Imagery has no CRS")
        if ds.count < 4: raise ValueError("Imagery must contain R,G,B,NIR bands")
        if ds.crs.to_string()!=TARGET_CRS: raise ValueError(f"Expected imagery in {TARGET_CRS}; got {ds.crs}")
        if min(abs(ds.transform.a),abs(ds.transform.e))>2.0: LOG.warning("Imagery resolution is coarser than 2 m: %s", ds.res)
    df=pd.read_csv(obs)
    required={"longitude","latitude","label","verification_status","observation_date","observation_id","positional_accuracy_m"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Observation CSV missing columns: {sorted(missing)}")
    if not set(df.label.dropna().unique()).issubset({0,1}): raise ValueError("label must contain only 0/1")
    if (df.verification_status.astype(str).str.lower().isin(["verified","independently_verified"]).sum()<1):
        raise ValueError("Require independently verified field labels; iNaturalist alone is not ground truth")
    return df

def metric_pack(y, p):
    pred=(p>=0.5).astype(int)
    return {"n":int(len(y)),"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),
        "recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
        "roc_auc":roc_auc_score(y,p) if len(np.unique(y))==2 else None,"mcc":matthews_corrcoef(y,pred),
        "brier":brier_score_loss(y,p),"confusion_matrix":confusion_matrix(y,pred).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery",type=Path,default=Path("data/buffalo_creek_naip.tif"))
    ap.add_argument("--observations",type=Path,default=Path("data/verified_observations.csv"))
    ap.add_argument("--output",type=Path,default=Path("outputs"))
    ap.add_argument("--buffer-m",type=float,default=45.0)
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    df=validate_inputs(args.imagery,args.observations)
    import geopandas as gpd
    points=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude,df.latitude),crs="EPSG:4326").to_crs(TARGET_CRS)
    args.output.mkdir(parents=True,exist_ok=True)
    with rasterio.open(args.imagery) as ds:
        raw=ds.read(masked=True).filled(np.nan)
        feat=raster_features(raw)
        rows=[]
        for geom in points.geometry:
            row,col=ds.index(geom.x,geom.y)
            if row<0 or col<0 or row>=ds.height or col>=ds.width: raise ValueError(f"Observation outside image: {geom}")
            rows.append(feat[:,row,col])
        X=np.asarray(rows).astype(float); y=df.label.to_numpy(int)
        valid=np.isfinite(X).all(axis=1)
        if valid.sum()<10 or len(np.unique(y[valid]))<2: raise ValueError("Insufficient valid labeled samples/classes")
        X=X[valid]; y=y[valid]; pts=points.loc[valid].reset_index(drop=True)
        # Spatial blocks are 1 km grid cells; buffer is applied geometrically to each test fold.
        groups=np.floor(pts.geometry.x/1000).astype(int).astype(str)+":"+np.floor(pts.geometry.y/1000).astype(int).astype(str)
        if groups.nunique()<2: raise ValueError("Need at least two spatial blocks")
        folds=min(5,groups.nunique()); gkf=GroupKFold(n_splits=folds); skf=StratifiedKFold(n_splits=min(5,int(np.bincount(y).min())),shuffle=True,random_state=SEED)
        model=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED)
        results={}
        for name, splitter, grouped in [("spatial",gkf,True),("stratified_baseline",skf,False)]:
            prob=np.full(len(y),np.nan); fold_metrics=[]
            splits=splitter.split(X,y,groups if grouped else None)
            for tr,te in splits:
                if grouped:
                    d=pts.geometry.iloc[tr].apply(lambda g: pts.geometry.iloc[te].distance(g).min()).to_numpy()
                    tr=tr[d>=args.buffer_m]
                if len(tr)==0 or len(np.unique(y[tr]))<2: raise ValueError(f"Fold has no usable training data after {args.buffer_m}m exclusion")
                m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED).fit(X[tr],y[tr])
                prob[te]=m.predict_proba(X[te])[:,1]; fold_metrics.append(metric_pack(y[te],prob[te]))
            if np.isnan(prob).any(): raise ValueError("Some samples received no out-of-fold prediction")
            results[name]={"overall":metric_pack(y,prob),"folds":fold_metrics}
        args.output.joinpath("metrics.json").write_text(json.dumps(results,indent=2))
        model.fit(X,y)
        # Windowed prediction preserves full raster georeferencing and bounds memory.
        profile=ds.profile.copy(); profile.update(count=1,dtype="float32",nodata=-9999,compress="deflate")
        with rasterio.open(args.output/"buckthorn_probability.tif","w",**profile) as out:
            for _, window in ds.block_windows(1):
                win=ds.read(window=window,masked=True).filled(np.nan); f=raster_features(win)
                matrix=f.reshape(len(FEATURES),-1).T; ok=np.isfinite(matrix).all(axis=1); pred=np.full(len(matrix),-9999,dtype=np.float32)
                if ok.any(): pred[ok]=model.predict_proba(matrix[ok])[:,1]
                out.write(pred.reshape(int(window.height),int(window.width)),1,window=window)
    pts.to_file(args.output/"observations.gpkg",driver="GPKG")
    LOG.info("Pipeline complete. Metrics are sample-based and not evidence of generalization beyond surveyed geography.")

if __name__=="__main__": main()
"""Reproducible buckthorn classifier. Requires real imagery and verified labels.

No synthetic or fallback data are generated. See README for input schema and caveats.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix)

LOG = logging.getLogger("buckthorn")
SEED = 2026
TARGET_CRS = "EPSG:26916"

def safe_index(a, b, offset=0.0):
    den = a + b + offset
    return np.divide(a-b, den, out=np.full_like(a, np.nan, dtype=float), where=np.abs(den)>1e-8)

def raster_features(arr):
    """Band-first uint/float NAIP with [R,G,B,NIR]; indices and 5/7 px texture."""
    if arr.ndim != 3 or arr.shape[0] < 4:
        raise ValueError("NAIP raster must have at least four bands in R,G,B,NIR order")
    r,g,b,nir = [arr[i].astype(float) for i in range(4)]
    ndvi=safe_index(nir,r); gndvi=safe_index(nir,g); grvi=safe_index(g,r); savi=1.5*safe_index(nir,r,0.5)
    def local_std(x,size):
        valid=np.isfinite(x); x0=np.nan_to_num(x)
        m=uniform_filter(x0,size=size); m2=uniform_filter(x0*x0,size=size)
        count=uniform_filter(valid.astype(float),size=size)
        return np.sqrt(np.maximum(np.divide(m2,count,out=np.full_like(m,np.nan),where=count>0)-np.square(np.divide(m,count,out=np.full_like(m,np.nan),where=count>0)),0))
    return np.stack([r,g,b,nir,ndvi,gndvi,grvi,savi,
        uniform_filter(np.nan_to_num(ndvi),size=5),local_std(ndvi,5),local_std(nir,7)])

FEATURES=["red","green","blue","nir","ndvi","gndvi","grvi","savi","ndvi_mean_5","ndvi_sd_5","nir_sd_7"]

def validate_inputs(tif, obs):
    if not tif.is_file() or not obs.is_file(): raise FileNotFoundError(f"Required inputs missing: {tif} and/or {obs}")
    with rasterio.open(tif) as ds:
        if ds.crs is None: raise ValueError("Imagery has no CRS")
        if ds.count < 4: raise ValueError("Imagery must contain R,G,B,NIR bands")
        if ds.crs.to_string()!=TARGET_CRS: raise ValueError(f"Expected imagery in {TARGET_CRS}; got {ds.crs}")
        if min(abs(ds.transform.a),abs(ds.transform.e))>2.0: LOG.warning("Imagery resolution is coarser than 2 m: %s", ds.res)
    df=pd.read_csv(obs)
    required={"longitude","latitude","label","verification_status","observation_date","observation_id","positional_accuracy_m"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Observation CSV missing columns: {sorted(missing)}")
    if not set(df.label.dropna().unique()).issubset({0,1}): raise ValueError("label must contain only 0/1")
    if (df.verification_status.astype(str).str.lower().isin(["verified","independently_verified"]).sum()<1):
        raise ValueError("Require independently verified field labels; iNaturalist alone is not ground truth")
    return df

def metric_pack(y, p):
    pred=(p>=0.5).astype(int)
    return {"n":int(len(y)),"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),
        "recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
        "roc_auc":roc_auc_score(y,p) if len(np.unique(y))==2 else None,"mcc":matthews_corrcoef(y,pred),
        "brier":brier_score_loss(y,p),"confusion_matrix":confusion_matrix(y,pred).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery",type=Path,default=Path("data/buffalo_creek_naip.tif"))
    ap.add_argument("--observations",type=Path,default=Path("data/verified_observations.csv"))
    ap.add_argument("--output",type=Path,default=Path("outputs"))
    ap.add_argument("--buffer-m",type=float,default=45.0)
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    df=validate_inputs(args.imagery,args.observations)
    import geopandas as gpd
    points=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude,df.latitude),crs="EPSG:4326").to_crs(TARGET_CRS)
    args.output.mkdir(parents=True,exist_ok=True)
    with rasterio.open(args.imagery) as ds:
        raw=ds.read(masked=True).filled(np.nan)
        feat=raster_features(raw)
        rows=[]
        for geom in points.geometry:
            row,col=ds.index(geom.x,geom.y)
            if row<0 or col<0 or row>=ds.height or col>=ds.width: raise ValueError(f"Observation outside image: {geom}")
            rows.append(feat[:,row,col])
        X=np.asarray(rows).astype(float); y=df.label.to_numpy(int)
        valid=np.isfinite(X).all(axis=1)
        if valid.sum()<10 or len(np.unique(y[valid]))<2: raise ValueError("Insufficient valid labeled samples/classes")
        X=X[valid]; y=y[valid]; pts=points.loc[valid].reset_index(drop=True)
        # Spatial blocks are 1 km grid cells; buffer is applied geometrically to each test fold.
        groups=np.floor(pts.geometry.x/1000).astype(int).astype(str)+":"+np.floor(pts.geometry.y/1000).astype(int).astype(str)
        if groups.nunique()<2: raise ValueError("Need at least two spatial blocks")
        folds=min(5,groups.nunique()); gkf=GroupKFold(n_splits=folds); skf=StratifiedKFold(n_splits=min(5,int(np.bincount(y).min())),shuffle=True,random_state=SEED)
        model=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED)
        results={}
        for name, splitter, grouped in [("spatial",gkf,True),("stratified_baseline",skf,False)]:
            prob=np.full(len(y),np.nan); fold_metrics=[]
            splits=splitter.split(X,y,groups if grouped else None)
            for tr,te in splits:
                if grouped:
                    d=pts.geometry.iloc[tr].apply(lambda g: pts.geometry.iloc[te].distance(g).min()).to_numpy()
                    tr=tr[d>=args.buffer_m]
                if len(tr)==0 or len(np.unique(y[tr]))<2: raise ValueError(f"Fold has no usable training data after {args.buffer_m}m exclusion")
                m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED).fit(X[tr],y[tr])
                prob[te]=m.predict_proba(X[te])[:,1]; fold_metrics.append(metric_pack(y[te],prob[te]))
            if np.isnan(prob).any(): raise ValueError("Some samples received no out-of-fold prediction")
            results[name]={"overall":metric_pack(y,prob),"folds":fold_metrics}
        args.output.joinpath("metrics.json").write_text(json.dumps(results,indent=2))
        model.fit(X,y)
        # Windowed prediction preserves full raster georeferencing and bounds memory.
        profile=ds.profile.copy(); profile.update(count=1,dtype="float32",nodata=-9999,compress="deflate")
        with rasterio.open(args.output/"buckthorn_probability.tif","w",**profile) as out:
            for _, window in ds.block_windows(1):
                win=ds.read(window=window,masked=True).filled(np.nan); f=raster_features(win)
                matrix=f.reshape(len(FEATURES),-1).T; ok=np.isfinite(matrix).all(axis=1); pred=np.full(len(matrix),-9999,dtype=np.float32)
                if ok.any(): pred[ok]=model.predict_proba(matrix[ok])[:,1]
                out.write(pred.reshape(int(window.height),int(window.width)),1,window=window)
    pts.to_file(args.output/"observations.gpkg",driver="GPKG")
    LOG.info("Pipeline complete. Metrics are sample-based and not evidence of generalization beyond surveyed geography.")

if __name__=="__main__": main()
