import numpy as np
import pytest

from buckthorn_spatial_pipeline import raster_features, safe_index, validate_inputs

def test_safe_index_marks_zero_denominator_invalid():
    out=safe_index(np.array([1.,0.]),np.array([-1.,0.]))
    assert np.isnan(out[0]) and np.isnan(out[1])

def test_feature_stack_has_expected_layers_and_shape():
    bands=np.ones((4,12,14),dtype=float)
    bands[3]=2
    result=raster_features(bands)
    assert result.shape==(11,12,14)
    assert np.isfinite(result).all()

def test_nodata_does_not_bias_texture_mean():
    bands=np.ones((4,11,11),dtype=float)
    bands[0]=100
    bands[3]=200
    bands[:,5,5]=np.nan
    result=raster_features(bands)
    assert result[8,5,5] == pytest.approx(1/3)

def test_missing_inputs_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_inputs(tmp_path/"absent.tif",tmp_path/"absent.csv")

