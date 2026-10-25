"""
CERIP — Tests: Grid Engine
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import geopandas as gpd
import pandas as pd
from src.data.grid_engine import build_grid, attach_features


def test_build_grid_returns_geodataframe():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0


def test_build_grid_cell_count():
    """Vizag bbox should produce ~1,700–2,000 cells at 500m."""
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    assert 3000 < len(gdf) < 4500, f"Expected ~3700 cells, got {len(gdf)}"


def test_build_grid_crs():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    assert gdf.crs.to_epsg() == 4326


def test_build_grid_centroid_columns():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    assert "centroid_lat" in gdf.columns
    assert "centroid_lon" in gdf.columns


def test_build_grid_lat_lon_in_bbox():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    assert gdf["centroid_lon"].between(83.10, 83.50).all()
    assert gdf["centroid_lat"].between(17.55, 17.90).all()


def test_attach_features_shape():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    fake_features = pd.DataFrame({"feature_a": range(len(gdf))})
    result = attach_features(gdf, fake_features)
    assert "feature_a" in result.columns
    assert len(result) == len(gdf)


def test_attach_features_fewer_than_grid():
    gdf = build_grid(83.15, 17.60, 83.45, 17.85, cell_size_m=500)
    small_feat = pd.DataFrame({"feature_x": range(100)})
    result = attach_features(gdf, small_feat)
    assert len(result) == len(gdf)
