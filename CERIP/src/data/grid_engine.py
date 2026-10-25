"""
CERIP — Grid Engine (M2)
=========================
Builds and manages the 500 m × 500 m spatial grid for Visakhapatnam.

Key functions
-------------
build_grid(bbox, cell_size_m, crs_proj, crs_geo)
    Create a GeoDataFrame of square grid cells covering the city bbox.

attach_features(grid_gdf, feature_df)
    Spatial join of tabular features onto the grid.

load_grid_geodataframe()
    Load the processed grid with all features as a GeoDataFrame.

Usage
-----
    from src.data.grid_engine import load_grid_geodataframe
    gdf = load_grid_geodataframe()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def build_grid(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    cell_size_m: int = 500,
    crs_proj: str = "EPSG:32644",  # UTM Zone 44N for metric calcs
    crs_geo: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Create a regular square grid of `cell_size_m` × `cell_size_m` cells
    covering the provided bounding box (in WGS84).

    Returns
    -------
    GeoDataFrame with columns:
        cell_id, geometry (WGS84 polygon), centroid_lon, centroid_lat,
        easting, northing  (projected, metres)
    """
    log.info("Building %dm grid over bbox (%.3f,%.3f)-(%.3f,%.3f) …",
             cell_size_m, lon_min, lat_min, lon_max, lat_max)

    # ── Project bbox corners to UTM ──────────────────────────────
    bbox_geo = gpd.GeoDataFrame(
        geometry=[box(lon_min, lat_min, lon_max, lat_max)],
        crs=crs_geo,
    ).to_crs(crs_proj)
    x_min, y_min, x_max, y_max = bbox_geo.total_bounds

    # ── Build grid in projected coordinates ─────────────────────
    xs = np.arange(x_min, x_max, cell_size_m)
    ys = np.arange(y_min, y_max, cell_size_m)

    cells = []
    cell_id = 0
    for x in xs:
        for y in ys:
            cells.append({
                "cell_id": cell_id,
                "easting":  x + cell_size_m / 2,   # centroid
                "northing": y + cell_size_m / 2,
                "geometry": box(x, y, x + cell_size_m, y + cell_size_m),
            })
            cell_id += 1

    gdf_proj = gpd.GeoDataFrame(cells, crs=crs_proj)
    gdf_geo  = gdf_proj.to_crs(crs_geo)

    # ── Add centroid lon/lat in WGS84 ───────────────────────────
    centroids = gdf_geo.geometry.centroid
    gdf_geo["centroid_lon"] = centroids.x.round(5)
    gdf_geo["centroid_lat"] = centroids.y.round(5)
    gdf_geo["easting"]      = gdf_proj["easting"].round(1)
    gdf_geo["northing"]     = gdf_proj["northing"].round(1)

    log.info("Grid built: %d cells", len(gdf_geo))
    return gdf_geo


def attach_features(
    grid_gdf: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Merge tabular features onto the grid via nearest-cell lookup
    (matches on lon/lat proximity when cell counts differ).

    In production this would be a proper spatial join with raster data.
    For the synthetic prototype, we match by index.
    """
    n_grid = len(grid_gdf)
    n_feat  = len(feature_df)

    if n_feat >= n_grid:
        # Sub-select to match grid size
        feat_subset = feature_df.iloc[:n_grid].reset_index(drop=True)
    else:
        # Tile features to fill grid (edge case)
        reps = (n_grid // n_feat) + 1
        feat_subset = pd.concat([feature_df] * reps, ignore_index=True).iloc[:n_grid]

    # Drop geometry-overlapping columns from feature_df if any
    cols_to_drop = [c for c in feat_subset.columns if c in ("cell_id", "geometry")]
    feat_subset = feat_subset.drop(columns=cols_to_drop)

    merged = grid_gdf.reset_index(drop=True).join(feat_subset)
    log.info("Features attached to grid. Final columns: %d", len(merged.columns))
    return merged


def load_grid_geodataframe(
    parquet_path: str | None = None,
    crs_proj: str = "EPSG:32644",
    crs_geo: str = "EPSG:4326",
    cell_size_m: int = 500,
) -> gpd.GeoDataFrame:
    """
    Load the full grid GeoDataFrame (build + attach features in one call).
    Requires synthetic data to have been generated first.

    Returns
    -------
    GeoDataFrame ready for ML training and map display.
    """
    import yaml

    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bbox    = cfg["city"]["bbox"]
    parquet_path = parquet_path or ROOT / "data" / "processed" / "grid_features.parquet"

    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. "
            "Run: python src/data/synthetic_generator.py"
        )

    feature_df = pd.read_parquet(parquet_path)

    grid_gdf = build_grid(
        lon_min=bbox["lon_min"],
        lat_min=bbox["lat_min"],
        lon_max=bbox["lon_max"],
        lat_max=bbox["lat_max"],
        cell_size_m=cell_size_m,
        crs_proj=crs_proj,
        crs_geo=crs_geo,
    )

    return attach_features(grid_gdf, feature_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    gdf = load_grid_geodataframe()
    print(gdf.head(3))
    print(f"\nTotal cells: {len(gdf):,}")
    print(f"Columns: {list(gdf.columns)}")
