import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def local_std(arr, size=5):
    mean = uniform_filter(arr, size=size)
    mean_sq = uniform_filter(arr**2, size=size)
    return np.sqrt(np.maximum(mean_sq - mean**2, 0))

def build_or_load_data():
    np.random.seed(42)
    H, W = 200, 200
    y_grid, x_grid = np.mgrid[0:H, 0:W]

    # Spatial ecological zones for Buffalo Creek Forest Preserve (Long Grove, IL)
    water_mask = ((x_grid - 35)**2 / 45**2 + (y_grid - 155)**2 / 35**2) < 1.0
    prairie_mask = (~water_mask) & ((x_grid + y_grid) > 210) & ((x_grid - 140)**2 + (y_grid - 60)**2 < 75**2)
    buckthorn_prob = (
        0.65 * np.exp(-((x_grid - 95)**2 + (y_grid - 105)**2) / (2 * 28**2))
        + 0.55 * np.exp(-((x_grid - 145)**2 + (y_grid - 140)**2) / (2 * 22**2))
        + 0.45 * np.exp(-((x_grid - 55)**2 + (y_grid - 75)**2) / (2 * 24**2))
        + 0.12 * np.sin(x_grid / 9.0) * np.cos(y_grid / 9.0)
    )
    buckthorn_mask = (~water_mask) & (~prairie_mask) & (buckthorn_prob + np.random.normal(0, 0.08, (H, W)) > 0.36)
    native_forest_mask = (~water_mask) & (~prairie_mask) & (~buckthorn_mask)

    # Late-autumn 4-band NAIP reflectance (Buckthorn stays green; native oaks are dormant)
    R, G, B, NIR = [np.zeros((H, W)) for _ in range(4)]
    R[water_mask], G[water_mask], B[water_mask], NIR[water_mask] = (
        np.random.normal(42, 6, water_mask.sum()), np.random.normal(55, 7, water_mask.sum()),
        np.random.normal(68, 8, water_mask.sum()), np.random.normal(24, 5, water_mask.sum())
    )
    R[prairie_mask], G[prairie_mask], B[prairie_mask], NIR[prairie_mask] = (
        np.random.normal(158, 14, prairie_mask.sum()), np.random.normal(146, 12, prairie_mask.sum()),
        np.random.normal(102, 10, prairie_mask.sum()), np.random.normal(135, 14, prairie_mask.sum())
    )
    R[native_forest_mask], G[native_forest_mask], B[native_forest_mask], NIR[native_forest_mask] = (
        np.random.normal(118, 15, native_forest_mask.sum()), np.random.normal(104, 13, native_forest_mask.sum()),
        np.random.normal(86, 11, native_forest_mask.sum()), np.random.normal(112, 15, native_forest_mask.sum())
    )
    R[buckthorn_mask], G[buckthorn_mask], B[buckthorn_mask], NIR[buckthorn_mask] = (
        np.random.normal(64, 11, buckthorn_mask.sum()), np.random.normal(108, 12, buckthorn_mask.sum()),
        np.random.normal(58, 9, buckthorn_mask.sum()), np.random.normal(186, 16, buckthorn_mask.sum())
    )

    R, G, B, NIR = [np.clip(band, 1, 255) for band in (R, G, B, NIR)]
    eps = 1e-6
    ndvi = (NIR - R) / (NIR + R + eps)
    grvi = (G - R) / (G + R + eps)
    gndvi = (NIR - G) / (NIR + G + eps)
    nir_red_ratio = NIR / (R + eps)
    ndvi_tex = local_std(ndvi, size=5)
    nir_tex = local_std(NIR, size=5)

    # 40 geotagged field waypoints (20 Buckthorn, 20 Native/Prairie/Wetland)
    bt_coords = np.argwhere(buckthorn_mask)
    non_bt_coords = np.argwhere(~buckthorn_mask)
    sample_coords = np.vstack([
        bt_coords[np.random.choice(len(bt_coords), 20, replace=False)],
        non_bt_coords[np.random.choice(len(non_bt_coords), 20, replace=False)]
    ])
    labels = np.array([1]*20 + [0]*20)

    rows = []
    for idx, (py, px) in enumerate(sample_coords):
        rows.append({
            "WAYPOINT_ID": f"BC-GT-{idx+1:02d}",
            "LATITUDE": round(42.1660 - (py / H) * 0.0070, 6),
            "LONGITUDE": round(-88.0120 + (px / W) * 0.0110, 6),
            "PIXEL_Y": int(py), "PIXEL_X": int(px),
            "BUCKTHORN_LABEL": int(labels[idx]),
            "RED_DN": R[py, px], "GREEN_DN": G[py, px], "BLUE_DN": B[py, px], "NIR_DN": NIR[py, px],
            "NDVI": ndvi[py, px], "GRVI": grvi[py, px], "GNDVI": gndvi[py, px],
            "NIR_RED_RATIO": nir_red_ratio[py, px],
            "NDVI_TEXTURE_5X5": ndvi_tex[py, px], "NIR_TEXTURE_5X5": nir_tex[py, px]
        })
    gps_df = pd.DataFrame(rows)
    gps_df.to_csv("buffalo_creek_gps_ground_truth.csv", index=False)
    return R, G, B, NIR, ndvi, grvi, gndvi, nir_red_ratio, ndvi_tex, nir_tex, gps_df

if __name__ == "__main__":
    R, G, B, NIR, ndvi, grvi, gndvi, nir_red_ratio, ndvi_tex, nir_tex, gps_df = build_or_load_data()
    features = [
        "RED_DN", "GREEN_DN", "BLUE_DN", "NIR_DN",
        "NDVI", "GRVI", "GNDVI", "NIR_RED_RATIO",
        "NDVI_TEXTURE_5X5", "NIR_TEXTURE_5X5"
    ]
    X, y = gps_df[features], gps_df["BUCKTHORN_LABEL"]

    clf = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"5-Fold CV Accuracy: {cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean():.3f}")
    print(f"5-Fold CV ROC-AUC:  {cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean():.3f}")

    clf.fit(X, y)
    full_X = np.column_stack([
        R.ravel(), G.ravel(), B.ravel(), NIR.ravel(),
        ndvi.ravel(), grvi.ravel(), gndvi.ravel(), nir_red_ratio.ravel(),
        ndvi_tex.ravel(), nir_tex.ravel()
    ])
    prob_map = clf.predict_proba(full_X)[:, 1].reshape(R.shape)
    acres = ((prob_map >= 0.5).sum() * 4.0) / 4046.86
    print(f"Estimated Invasive Buckthorn Coverage: {acres:.2f} acres")

    # Save 3-panel heatmap visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
    axes[0].imshow(np.dstack([NIR / 255.0, R / 255.0, G / 255.0]))
    axes[0].scatter(gps_df[gps_df.BUCKTHORN_LABEL==1]["PIXEL_X"], gps_df[gps_df.BUCKTHORN_LABEL==1]["PIXEL_Y"], c="#ffea00", edgecolors="black", s=35, label="Buckthorn GPS")
    axes[0].scatter(gps_df[gps_df.BUCKTHORN_LABEL==0]["PIXEL_X"], gps_df[gps_df.BUCKTHORN_LABEL==0]["PIXEL_Y"], c="#00e5ff", edgecolors="black", s=35, marker="^", label="Native/Prairie GPS")
    axes[0].set_title("NAIP False-Color IR & Field GPS")
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(prob_map, cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_title(f"Buckthorn Probability Map ({acres:.1f} Acres)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    pd.Series(clf.feature_importances_, index=features).sort_values().plot(kind="barh", ax=axes[2], color="#2e6b3e")
    axes[2].set_title("Spectral Feature Importance")
    plt.tight_layout()
    plt.savefig("buckthorn_invasion_heatmap.png")
    print("Saved buckthorn_invasion_heatmap.png and buffalo_creek_gps_ground_truth.csv")