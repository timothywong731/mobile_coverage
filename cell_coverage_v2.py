from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from datasets import load_dataset
from dateutil import rrule
from keplergl import KeplerGl
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from shapely.validation import make_valid
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

START_DATE = datetime(2025, 3, 1)
END_DATE = datetime(2025, 6, 30)


def month_sequence(start: datetime, end: datetime) -> list[str]:
    """Return ISO-formatted month start dates spanning the inclusive range."""

    return [
        month.strftime("%Y-%m-%d")
        for month in rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)
    ]


DEFAULT_MONTHS = month_sequence(START_DATE, END_DATE)

DEFAULT_DATA_FILES = [
    "np_extract_part_1.csv",
    "np_extract_part_2.csv",
    "np_extract_part_3.csv",
    "np_extract_part_4.csv",
    "np_extract_part_5.csv",
    "np_extract_part_6.csv",
    # "np_extract_part_7.csv",
    # "np_extract_part_8.csv",
    # "np_extract_part_9.csv",
]

OUTLIER_CELLS = {"4dc7c9ec434ed06502767136789763ec11d2c4b7"}

DEFAULT_CELL_IDS = [
    "b6b78eec6eb1e85d7b0de7fb49d12c4aadcd3b1b",
    "83c3d7642c3848655ad61f0ed3877799d3c75074",
    "f117748b1c42637547c239987a174cfc31af8f61",
    "1b6d62dd71f3b16b97d1524dc6161a0dff3873a7",
    "88177bc3f4230c4ff34ef191b4af0fbe0fea8501",
    "901c6ee62216324f15817d992a0b9dd89789760c",
]

MIN_POINTS_REQUIRED = 1


def load_measurements(data_files: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load raw measurement data from the Hugging Face dataset or local CSV shards."""

    files = list(data_files) if data_files else DEFAULT_DATA_FILES
    dataset = load_dataset(
        "joefee/cell-service-data",
        data_files={"train": files},
    )
    return dataset["train"].to_pandas()


def prepare_signal_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical signal strength buckets aligned with Ofcom guidance."""

    categorised = df.copy()
    categorised["signal_level_category"] = pd.cut(
        categorised["signal_level"],
        bins=[-np.inf, -105, -95, -82, -74, np.inf],
        labels=["1. Very Weak", "2. Weak", "3. Moderate", "4. Strong", "5. Very Strong"],
    )
    return categorised


def get_sufficient_data_cells(
    df: pd.DataFrame,
    months: Sequence[str],
    min_points_required: int = MIN_POINTS_REQUIRED,
) -> list[str]:
    """Return cell IDs with at least ``min_points_required`` samples per active month."""

    if df.empty:
        return []

    con = duckdb.connect()
    try:
        con.register("df", df)
        query = f"""
WITH monthly_count AS (
    SELECT
        unique_cell,
        date_trunc('month', CAST(timestamp AS timestamp)) AS month,
        COUNT(*) AS count
    FROM df
    WHERE signal_level IS NOT NULL
    GROUP BY unique_cell, month
    HAVING COUNT(*) >= {min_points_required}
)
SELECT unique_cell
FROM monthly_count
GROUP BY unique_cell
-- HAVING COUNT(DISTINCT month) = {len(months)}
"""
        result = con.execute(query).fetchall()
    finally:
        con.close()

    return [row[0] for row in result if row[0] not in OUTLIER_CELLS]


def generate_convex_hull_geom(df, quantile: float = 0.95) -> BaseGeometry:
    """
    Generate a valid convex hull MultiPolygon from input DataFrame.
    Converts input longitude/latitude columns to float type if they are not
    already.

    Args:
        df (pd.DataFrame): DataFrame with 'longitude' and 'latitude' columns.
        **args: Currently unused, present for API consistency.
    Returns:
        BaseGeometry:
            Shapely geometry object representing the convex hull,
            or None if unsuccessful.
    """

    # Construct a convex hull with shapely using train_df
    points_train = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    # Find center of mass among these points
    multipoint = MultiPoint(points_train)
    center_of_mass = multipoint.centroid

    # For each point, calculate distance to center of mass
    distances = [point.distance(center_of_mass) for point in points_train]

    # Find the 95% percentile distance
    threshold_distance = pd.Series(distances).quantile(quantile)

    # Filter points to only those within the threshold distance
    filtered_points = [
        point for point, distance in zip(
            points_train, distances) if distance <= threshold_distance]

    # Create new multipoint from filtered points
    multipoint_filtered = MultiPoint(filtered_points)

    # Calculate the convex hull
    convex_hull = multipoint_filtered.convex_hull

    if convex_hull.is_valid:
        return convex_hull
    else:
        print("Warning: Convex hull is invalid.")
        return None


def generate_svm_boundary_geom(df, **args) -> BaseGeometry:
    """
    Generate a valid MultiPolygon with true cut-out holes from One-Class SVM
    boundary. Converts input longitude/latitude columns to float type if they
    are not already.

    Args:
        df (pd.DataFrame): DataFrame with 'longitude' and 'latitude' columns.
                        These columns can contain numbers or Decimal objects.
        **args: Keyword arguments passed directly to svm.OneClassSVM.

    Returns:
        BaseGeometry:
            A valid MultiPolygon geometry representing the SVM boundary,
            or None if the boundary could not be created.
    """

    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['longitude', 'latitude']):
        print("Error: Input df must be a pandas DataFrame with 'longitude' and 'latitude' columns.")
        return None

    if len(df) < 2:
        print("Warning: Need at least 2 data points for SVM.")
        return None

    # --- Convert coordinate columns to float type ---
    # This resolves the Decimal vs float TypeError
    try:
        df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame
        df_copy['longitude'] = df_copy['longitude'].astype(float)
        df_copy['latitude'] = df_copy['latitude'].astype(float)
        coords = df_copy[['longitude', 'latitude']].values
    except (TypeError, ValueError) as e:
        print(f"Error converting coordinate columns to float: {e}")
        return None

    # --- 1. Train the SVM ---
    try:
        clf = svm.OneClassSVM(**args)
        clf.fit(coords)
    except Exception as e:
        print(f"Error during SVM training: {e}")
        return None

    # --- 2. Create mesh grid for contouring ---
    # Now calculations will use standard floats
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 1 if x_range > 1e-9 else 0.1
    y_margin = y_range * 1 if y_range > 1e-9 else 0.1

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    resolution = 500
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    try:
        Z = clf.decision_function(grid_points).reshape(xx.shape)
    except Exception as e:
        print(f"Error during SVM decision function evaluation: {e}")
        return None

    # --- 3. Extract contour lines at level 0 (the boundary) ---
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xx, yy, Z, levels=[0])
    except Exception as e:
        print(f"Error during contour generation: {e}")
        plt.close(fig)
        return None
    plt.close(fig)

    if not cs.allsegs or not cs.allsegs[0]:
        print("Warning: No contour lines found at level 0.")
        return None

    segments = cs.allsegs[0]
    lines = [LineString(seg) for seg in segments if len(seg) >= 2]

    if not lines:
        print("Warning: No valid LineStrings created from contour segments.")
        return None

    # --- 4. Polygonize the lines ---
    try:
        all_polygons = list(polygonize(lines))
    except Exception as e:
        print(f"Error during polygonization: {e}")
        return None

    if not all_polygons:
        print("Warning: Polygonization did not yield any polygons.")
        return None

    # --- 5. Classify polygons and perform unary union ---
    positive_polygons = []
    for p in all_polygons:
        if p.is_valid and p.area > 1e-9:
            rep_point = p.representative_point()
            try:
                decision_val = clf.decision_function([[rep_point.x, rep_point.y]])[0]
                if decision_val >= 0:
                    # Ensure the polygon added is valid - unary_union can struggle with invalid inputs
                    if p.is_valid:
                        positive_polygons.append(p)
                    else:
                    # Attempt to buffer by 0 to fix potential self-intersections
                        buffered_p = p.buffer(0)
                        if buffered_p.is_valid and isinstance(buffered_p, Polygon):
                                positive_polygons.append(buffered_p)
                        else:
                            print(f"Warning: Skipping invalid polygon generated during classification step even after buffer(0). Area: {p.area}")

            except Exception as e:
                print(f"Warning: Error checking decision function for a polygon point: {e}")


    if not positive_polygons:
        print("Warning: No valid polygons were classified as inside the SVM boundary.")
        return None

    # --- 6. Unary Union to merge positive polygons and create holes ---
    try:
        # Filter again for validity just before union, as buffer(0) might create MultiPolygons
        valid_positive_polygons = [poly for poly in positive_polygons if poly.is_valid and isinstance(poly, Polygon)]
        if not valid_positive_polygons:
            print("Warning: No valid polygons remaining before unary union.")
            return None
        result_geom = unary_union(valid_positive_polygons)

    except Exception as e:
        # Catch potential errors during unary_union (often related to complex topology)
        print(f"Error during unary union: {e}")
        # As a fallback, try creating a MultiPolygon directly from the valid positive polygons
        # This might result in overlaps instead of proper union, but is better than nothing.
        print("Attempting fallback: creating MultiPolygon from individual positive polygons.")
        try:
            result_geom = MultiPolygon(valid_positive_polygons)
            if not result_geom.is_valid:
                print("Warning: Fallback MultiPolygon is invalid.")
                # Try buffer(0) on the multipolygon as a last resort
                buffered_result = result_geom.buffer(0)
                if buffered_result.is_valid:
                    print("Fallback MultiPolygon fixed with buffer(0).")
                    result_geom = buffered_result
                else:
                    print("Error: Fallback MultiPolygon remains invalid even after buffer(0). Cannot proceed.")
                    return None
        except Exception as fallback_e:
            print(f"Error during fallback MultiPolygon creation: {fallback_e}")
            return None


    # --- 7. Format output as MultiPolygon GeoJSON mapping ---
    final_multi_poly = None
    if result_geom is None: # Should not happen with current logic, but check anyway
        print("Error: Resulting geometry is None after union/fallback.")
        return None

    # Simplify handling by ensuring result_geom is always iterable (list of polygons)
    geoms_to_wrap = []
    if isinstance(result_geom, Polygon):
        if result_geom.is_valid:
            geoms_to_wrap = [result_geom]
    elif isinstance(result_geom, MultiPolygon):
        # Filter out invalid geoms within the MultiPolygon if any
        geoms_to_wrap = [g for g in result_geom.geoms if g.is_valid and isinstance(g, Polygon)]
    elif hasattr(result_geom, 'geoms'): # Handle GeometryCollection
        print("Warning: unary_union resulted in a GeometryCollection. Filtering for valid Polygons.")
        geoms_to_wrap = [g for g in result_geom.geoms if g.is_valid and isinstance(g, Polygon)]

    if not geoms_to_wrap:
        print("Warning: No valid polygons found in the final geometry after union/cleanup.")
        return None

    # Create the final MultiPolygon
    final_multi_poly = MultiPolygon(geoms_to_wrap)

    # Final validity check
    if final_multi_poly.is_valid:
        return final_multi_poly
    else:
        # Try one last buffer(0) fix
        print("Warning: Final MultiPolygon is invalid. Attempting buffer(0) fix.")
        buffered_final = final_multi_poly.buffer(0)
        if buffered_final.is_valid and isinstance(buffered_final, (Polygon, MultiPolygon)):
            # Re-wrap if buffer resulted in a single Polygon
            if isinstance(buffered_final, Polygon):
                final_multi_poly = MultiPolygon([buffered_final])
            else:
                final_multi_poly = buffered_final
            print("Final MultiPolygon fixed with buffer(0).")
            return final_multi_poly
        else:
            print("Error: Final MultiPolygon remains invalid even after buffer(0).")
            return None


def generate_context_svm_boundary_geom(
    aoi_gdf: gpd.GeoDataFrame,
    featurizer,                          # callable: (N,2) lon/lat -> (X, feature_cols)
    scaler: StandardScaler,
    clf: svm.OneClassSVM,
    resolution: int = 300,
    min_ring_pts: int = 3,
) -> Optional[BaseGeometry]:

    # --- 1) AOI to WGS84 and grid its bbox ---
    aoi_wgs = aoi_gdf.to_crs(4326)
    if len(aoi_wgs) == 0 or aoi_wgs.iloc[0].geometry is None:
        return None
    minx, miny, maxx, maxy = aoi_wgs.total_bounds

    # guard degenerate bbox
    if not np.isfinite([minx, miny, maxx, maxy]).all() or (maxx - minx) <= 0 or (maxy - miny) <= 0:
        return None

    gx, gy = np.meshgrid(
        np.linspace(minx, maxx, resolution),
        np.linspace(miny, maxy, resolution)
    )
    grid_lonlat = np.c_[gx.ravel(), gy.ravel()]

    # --- 2) Build grid features with SAME featurizer and scale them ---
    Xg, _ = featurizer(grid_lonlat)             # shape: (N, n_features)
    Xg = scaler.transform(Xg)

    # --- 3) OC-SVM scores over grid ---
    Z = clf.decision_function(Xg).reshape(gx.shape)

    # --- 4) 0-level contour and polygonization ---
    # We use matplotlib's contour for isolines, but immediately close the figure.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(gx, gy, Z, levels=[0.0])
    finally:
        plt.close(fig)

    if not cs.allsegs or not cs.allsegs[0]:
        return None

    # Convert contour segments to LineStrings (filter very short ones)
    segs = [seg for seg in cs.allsegs[0] if len(seg) >= min_ring_pts]
    if not segs:
        return None
    lines = [LineString(seg) for seg in segs]

    # Polygonize the closed lines; union & clean validity
    polys = list(polygonize(lines))
    if not polys:
        return None

    merged = unary_union(polys)
    # Clean possible self-touching rings
    try:
        merged = make_valid(merged)
    except Exception:
        pass

    # Clip to AOI (optional but usually desirable)
    try:
        aoi_poly = aoi_wgs.iloc[0].geometry
        merged = merged.intersection(aoi_poly)
    except Exception:
        # if clip fails, return un-clipped
        pass

    # Final guard
    if merged is None or merged.is_empty:
        return None

    return merged


def get_classification_metric(y_true, y_pred, metric: str):
    """
    Compute a specific classification metric based on the provided metric name.
    Handles zero division by returning 0 for precision, recall, and F1 score in such cases.

    Args:
        y_true (list): True binary labels.
        y_pred (list): Predicted binary labels.
        metric (str): Metric to compute - one of 'accuracy', 'precision', 'recall', 'f1'.

    Returns:
        float: The computed metric score.
    """
    metric = metric.lower()
    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == 'recall':
        return recall_score(y_true, y_pred, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, zero_division=0)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'accuracy', 'precision', 'recall', 'f1'.")


def _to_work_crs(gdf: gpd.GeoDataFrame, work_crs: int | str) -> gpd.GeoDataFrame:
    """
    Project a GeoDataFrame to the working CRS and repair invalid geometries.
    """
    if gdf.crs is None:
        # Assume WGS84 if missing (osmnx usually returns WGS84)
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf.to_crs(work_crs)
    if "geometry" in gdf.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def fetch_osm_by_hull(
    hull_geom,                         # shapely Polygon/MultiPolygon
    hull_crs: int | str = 4326,
    work_crs: int | str = 27700,
    buffer_m: float = 200.0,
    building_tags: Optional[Dict[str, Any]] = None,
    road_network_type: str = "drive",  # "drive", "walk", "all", etc.
    road_simplify: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetch OSM buildings & roads inside a buffered AOI derived from hull_geom.
    Returns (buildings_gdf, roads_gdf, aoi_gdf) all in work_crs.
    """
    # 1) Prepare AOI in working CRS, buffer in meters, clean & dissolve
    aoi_work = gpd.GeoSeries([hull_geom], crs=hull_crs).to_crs(work_crs)
    if buffer_m and buffer_m != 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aoi_work = aoi_work.buffer(buffer_m)
    aoi_work = aoi_work.buffer(0)
    aoi_work = gpd.GeoSeries([unary_union(aoi_work)], crs=work_crs)

    # WGS84 for OSM queries
    aoi_wgs = aoi_work.to_crs(4326).iloc[0]

    # 2) Default building tags
    if building_tags is None:
        building_tags = {"building": True}

    # 3) Fetch buildings
    try:
        bldg = ox.features_from_polygon(aoi_wgs, tags=building_tags)
        bldg = bldg[bldg.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception as e:
        warnings.warn(f"OSM building fetch failed: {e}")
        bldg = gpd.GeoDataFrame(geometry=[], crs=4326)

    # 4) Fetch roads
    try:
        G = ox.graph_from_polygon(aoi_wgs, network_type=road_network_type, simplify=road_simplify)
        roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
        roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    except Exception as e:
        warnings.warn(f"OSM road fetch failed: {e}")
        roads = gpd.GeoDataFrame(geometry=[], crs=4326)

    # 5) Reproject to work CRS and clip to AOI precisely
    aoi_gdf = gpd.GeoDataFrame({"geometry": [aoi_work.iloc[0]]}, crs=work_crs)

    bldg = _to_work_crs(bldg, work_crs)
    roads = _to_work_crs(roads, work_crs)

    try:
        if len(bldg):
            bldg = gpd.clip(bldg, aoi_gdf)
    except Exception as e:
        warnings.warn(f"Clip buildings failed: {e}")

    try:
        if len(roads):
            roads = gpd.clip(roads, aoi_gdf)
    except Exception as e:
        warnings.warn(f"Clip roads failed: {e}")

    # Tidy minimal columns
    bldg = bldg.rename(columns=lambda c: str(c)).reset_index(drop=True)
    roads = roads.rename(columns=lambda c: str(c)).reset_index(drop=True)

    return bldg, roads, aoi_gdf


def prep_osm_layers(
    bldg_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    work_crs: int | str = 27700
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Ensure layers are in a metric CRS, clean geometry types, fix invalids, and clip to AOI.
    Returns (buildings, roads, aoi) all in work_crs.
    """
    bldg = bldg_gdf.to_crs(work_crs).copy()
    roads = roads_gdf.to_crs(work_crs).copy()
    aoi   = aoi_gdf.to_crs(work_crs).copy()

    # Buildings: polygons only + repair
    bldg = bldg[bldg.geometry.notna() & bldg.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bldg["geometry"] = bldg.geometry.buffer(0)

    # Roads: lineal only; explode multilines; repair
    roads = roads[roads.geometry.notna() & roads.geometry.type.isin(["LineString","MultiLineString"])].copy()
    # geopandas >=0.10: explode exists
    if hasattr(roads, "explode"):
        roads = roads.explode(index_parts=False, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roads["geometry"] = roads.geometry.buffer(0)

    # Clip to AOI
    try:
        bldg = gpd.clip(bldg, aoi)
        roads = gpd.clip(roads, aoi)
    except Exception:
        pass

    return bldg.reset_index(drop=True), roads.reset_index(drop=True), aoi.reset_index(drop=True)


def build_context_pack(
    bldg: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    radii_m: List[int] = [100]
) -> Dict[str, Any]:
    """
    Build reusable context objects:
      - dissolved building union (polygon)
      - STRtree for roads
      - feature column schema
    """
    # Building union
    b_union = None
    if len(bldg):
        try:
            u = unary_union(bldg.geometry.values)
        except Exception:
            u = unary_union(bldg.geometry.buffer(0).values)
        b_union = u.buffer(0)

    # Road index
    road_geoms = list(roads.geometry.values) if len(roads) else []
    road_tree = STRtree(road_geoms) if road_geoms else None

    feature_cols = (
        ["x", "y", "inside_bldg", "dist_bldg_edge"]
        + [f"bldg_area_r{r}" for r in radii_m]
        + [f"road_len_r{r}" for r in radii_m]
    )

    return {
        "b_union": b_union,
        "road_tree": road_tree,
        "road_geoms": road_geoms,
        "radii_m": radii_m,
        "feature_cols": feature_cols,
        "work_crs": bldg.crs,
    }


def make_featurizer(
    context_pack: Dict[str, Any],
    max_edge_dist: float = 500.0,
    batch: int = 5000
):
    """
    Return a callable: featurizer(lonlat_np) -> (X, feature_cols)
    lonlat_np columns: [lon, lat] in WGS84.
    """
    b_union    = context_pack["b_union"]
    road_tree  = context_pack["road_tree"]
    road_geoms = context_pack["road_geoms"]
    radii_m    = context_pack["radii_m"]
    work_crs   = context_pack["work_crs"]
    feature_cols = context_pack["feature_cols"]

    def featurize_lonlat(lonlat_np: np.ndarray):
        # Build points GDF in work CRS
        gdf = gpd.GeoDataFrame(
            {"lon": lonlat_np[:, 0], "lat": lonlat_np[:, 1]},
            geometry=gpd.points_from_xy(lonlat_np[:, 0], lonlat_np[:, 1]),
            crs=4326
        ).to_crs(work_crs).reset_index(drop=True)

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        # Inside-building & distance to building edge
        if b_union is not None and not getattr(b_union, "is_empty", False):
            try:
                gdf["inside_bldg"] = gdf.geometry.within(b_union).astype(int)
            except GEOSException:
                gdf["inside_bldg"] = gdf.geometry.buffer(0).within(b_union).astype(int)
            # distance to union boundary; 0 if inside
            d = gdf.geometry.apply(lambda p: 0.0 if p.within(b_union) else p.distance(b_union.boundary))
            gdf["dist_bldg_edge"] = np.minimum(d.values, max_edge_dist)
        else:
            gdf["inside_bldg"] = 0
            gdf["dist_bldg_edge"] = max_edge_dist

        # Ring features
        for r in radii_m:
            area_norm = np.pi * (r ** 2)
            b_vals = np.zeros(len(gdf), dtype=float)
            rd_vals = np.zeros(len(gdf), dtype=float)

            # batch buffers to control memory
            for s in range(0, len(gdf), batch):
                e = min(s + batch, len(gdf))
                bufs = [pt.buffer(r) for pt in gdf.geometry.iloc[s:e]]

                # building area density
                if b_union is not None:
                    b_areas = []
                    for bf in bufs:
                        try:
                            inter = bf.intersection(b_union)
                            b_areas.append(inter.area if not inter.is_empty else 0.0)
                        except GEOSException:
                            b_areas.append(0.0)
                    b_vals[s:e] = np.asarray(b_areas) / area_norm

                # road length density
                if road_tree is not None:
                    lens = []
                    for bf in bufs:
                        cand = road_tree.query(bf)
                        if cand is None or (hasattr(cand, "size") and cand.size == 0) or (hasattr(cand, "__len__") and len(cand) == 0):
                            lens.append(0.0)
                            continue
                        total = 0.0
                        # cand may be geoms or indices depending on Shapely version
                        iterable = (cand if hasattr(cand[0], "intersection") else (road_geoms[i] for i in cand))
                        for seg in iterable:
                            try:
                                total += seg.intersection(bf).length
                            except GEOSException:
                                total += seg.buffer(0).intersection(bf).length
                        lens.append(total / area_norm)
                    rd_vals[s:e] = np.asarray(lens)

            gdf[f"bldg_area_r{r}"] = b_vals
            gdf[f"road_len_r{r}"] = rd_vals

        return gdf[feature_cols].to_numpy(), feature_cols

    return featurize_lonlat


def build_context_from_polygon(
    aoi_geom,                    # shapely Polygon/MultiPolygon (e.g., your wide convex hull)
    aoi_crs: int | str = 4326,   # CRS of aoi_geom (WGS84 by default)
    work_crs: int | str = 27700, # metric CRS for geometry ops (BNG here)
    buffer_m: float = 200.0,     # expand AOI slightly before querying OSM
    building_tags: dict | None = None,
    road_network_type: str = "drive",
    road_simplify: bool = True,
    radii_m: tuple[int, ...] = (100,),  # ring radii for density features
    max_edge_dist: float = 500.0,       # cap on road-edge distance
    batch: int = 4000                   # batching for buffers/intersections
):
    """
    From a large polygon (AOI), fetch OSM context and build a featurizer.

    Returns:
      bldg_gdf     : GeoDataFrame (work_crs) of building polygons
      roads_gdf    : GeoDataFrame (work_crs) of road lines
      aoi_gdf      : GeoDataFrame (work_crs) single-row AOI polygon
      feature_cols : list[str] feature column names (stable order)
      featurizer   : callable (lonlat_np) -> (X, feature_cols)
                     lonlat_np shape: (N, 2) with columns [lon, lat] (WGS84)
    """
    # 1) Pull OSM layers for the AOI
    bldg_raw, roads_raw, aoi = fetch_osm_by_hull(
        hull_geom=aoi_geom,
        hull_crs=aoi_crs,
        work_crs=work_crs,
        buffer_m=buffer_m,
        building_tags=building_tags,
        road_network_type=road_network_type,
        road_simplify=road_simplify,
    )

    # 2) Normalize/repair layers in work_crs
    bldg_gdf, roads_gdf, aoi_gdf = prep_osm_layers(bldg_raw, roads_raw, aoi, work_crs=work_crs)

    # 3) Build context pack and featurizer
    ctx_pack = build_context_pack(bldg_gdf, roads_gdf, radii_m=list(radii_m))
    featurizer = make_featurizer(ctx_pack, max_edge_dist=max_edge_dist, batch=batch)
    feature_cols = ctx_pack["feature_cols"]

    return bldg_gdf, roads_gdf, aoi_gdf, feature_cols, featurizer


class ContextBuilder:
    def __init__(self, aoi_geom, aoi_crs=4326, work_crs=27700, buffer_m=200.0):
        self.aoi_geom = aoi_geom
        self.aoi_crs = aoi_crs
        self.work_crs = work_crs
        self.buffer_m = buffer_m

        # Registered layers and active toggles
        self._layer_registry = {}
        self._active_layers = {}

        # Context params
        self._radii_m = [100]
        self._max_edge_dist = 500.0
        self._batch = 4000

        # Auto-register core OSM layers
        self.register_layer("buildings", self._fetch_buildings)
        self.register_layer("roads", self._fetch_roads)

    # ------------------------------------------------------
    # 📦 Layer registration system
    # ------------------------------------------------------
    def register_layer(self, name, func):
        """Register a callable that fetches/prepares a GeoDataFrame for this AOI."""
        self._layer_registry[name] = func
        return self

    def with_layer(self, name, enabled=True, **kwargs):
        """Toggle a layer by name and pass layer-specific kwargs."""
        if name not in self._layer_registry:
            raise ValueError(f"Unknown layer: {name}")
        self._active_layers[name] = {"enabled": enabled, "kwargs": kwargs}
        return self

    def with_rings(self, radii_m):
        self._radii_m = list(radii_m)
        return self

    def with_max_edge_dist(self, val):
        self._max_edge_dist = val
        return self

    def with_batch(self, val):
        self._batch = val
        return self

    # ------------------------------------------------------
    # 🧩 Default layer implementations
    # ------------------------------------------------------
    def _fetch_buildings(self, aoi_wgs, **kwargs):
        tags = kwargs.get("tags", {"building": True})
        try:
            bldg = ox.features_from_polygon(aoi_wgs, tags=tags)
            bldg = bldg[bldg.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        except Exception as e:
            warnings.warn(f"[buildings] fetch failed: {e}")
            bldg = gpd.GeoDataFrame(geometry=[], crs=4326)
        return bldg

    def _fetch_roads(self, aoi_wgs, **kwargs):
        ntype = kwargs.get("network_type", "drive")
        simplify = kwargs.get("simplify", True)
        try:
            G = ox.graph_from_polygon(aoi_wgs, network_type=ntype, simplify=simplify)
            roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        except Exception as e:
            warnings.warn(f"[roads] fetch failed: {e}")
            roads = gpd.GeoDataFrame(geometry=[], crs=4326)
        return roads

    # Example extra module: greenspace
    def _fetch_greenspace(self, aoi_wgs, **kwargs):
        tags = {"leisure": ["park", "garden"], "landuse": "grass"}
        try:
            gs = ox.features_from_polygon(aoi_wgs, tags=tags)
            gs = gs[gs.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        except Exception as e:
            warnings.warn(f"[greenspace] fetch failed: {e}")
            gs = gpd.GeoDataFrame(geometry=[], crs=4326)
        return gs

    # Example extra module: water
    def _fetch_water(self, aoi_wgs, **kwargs):
        tags = {"natural": ["water", "wetland"]}
        try:
            w = ox.features_from_polygon(aoi_wgs, tags=tags)
            w = w[w.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        except Exception as e:
            warnings.warn(f"[water] fetch failed: {e}")
            w = gpd.GeoDataFrame(geometry=[], crs=4326)
        return w

    # ------------------------------------------------------
    # 🏗 Build full context
    # ------------------------------------------------------
    def build(self):
        # Project AOI
        aoi_work = gpd.GeoSeries([self.aoi_geom], crs=self.aoi_crs).to_crs(self.work_crs)
        aoi_wgs = aoi_work.to_crs(4326).iloc[0]

        # Fetch all active layers
        layers = {}
        for lname, config in self._active_layers.items():
            if not config["enabled"]:
                continue
            func = self._layer_registry[lname]
            layers[lname] = func(aoi_wgs, **config["kwargs"])

        # Merge & normalize
        layers = {k: v.to_crs(self.work_crs) for k, v in layers.items()}

        # Build context pack and featurizer
        ctx_pack = build_context_pack(
            bldg=layers.get("buildings", gpd.GeoDataFrame(geometry=[], crs=self.work_crs)),
            roads=layers.get("roads", gpd.GeoDataFrame(geometry=[], crs=self.work_crs)),
            radii_m=self._radii_m
        )

        featurizer = make_featurizer(ctx_pack, max_edge_dist=self._max_edge_dist, batch=self._batch)
        feature_cols = ctx_pack["feature_cols"]

        # Keep reference to layers in ctx_pack for inspection
        ctx_pack.update({f"{k}_gdf": v for k, v in layers.items()})
        ctx_pack["aoi_gdf"] = gpd.GeoDataFrame({"geometry": [aoi_work.iloc[0]]}, crs=self.work_crs)

        print(f"Built context with layers: {list(layers.keys())}")
        return ctx_pack, featurizer, feature_cols


def featurize_points_df(df_points, featurizer, lon_col="longitude", lat_col="latitude"):
    """
    Convert a DataFrame with lon/lat columns to context features using the featurizer.
    Returns: X (np.ndarray), feature_cols (list[str])
    """
    lonlat = df_points[[lon_col, lat_col]].to_numpy()
    X, feature_cols = featurizer(lonlat)
    return X, feature_cols


def _truth_polygon_from_points(df_true_pts: pd.DataFrame,
                               work_crs: int | str = 27700,
                               method: str = "hull",
                               buffer_m: float = 0.0):
    """
    Build a ground-truth polygon from positives (this cell in target levels).
    method: 'hull' (convex hull) or 'alpha' (very light concavity via triangulation).
    """
    if df_true_pts.empty:
        return None

    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df_true_pts["longitude"], df_true_pts["latitude"]),
        crs=4326
    ).to_crs(work_crs)

    if len(gdf) < 3:
        # not enough to form polygon
        return None

    if method == "alpha":
        # lightweight alpha-shape (falls back to hull if unstable)
        try:
            from math import dist
            mp = MultiPoint(list(gdf.geometry.values))
            tris = list(shapely.ops.triangulate(mp))
            if not tris:
                geom = mp.convex_hull
            else:
                # pick alpha via heuristic on median edge length
                edges = []
                for t in tris:
                    a,b,c = t.exterior.coords[:3]
                    edges += [dist(a,b), dist(b,c), dist(c,a)]
                med = np.median(edges) if edges else 1.0
                alpha = max(1e-9, 3.0/med)  # heuristic
                inv_a = 1.0/alpha
                keep = []
                for tri in tris:
                    a,b,c = tri.exterior.coords[:3]
                    la, lb, lc = dist(a,b), dist(b,c), dist(c,a)
                    s = 0.5*(la+lb+lc)
                    area2 = max(s*(s-la)*(s-lb)*(s-lc), 0.0)
                    if area2 == 0:
                        continue
                    R = (la*lb*lc)/(4.0*np.sqrt(area2))
                    if R <= inv_a:
                        keep.append(tri)
                geom = unary_union(keep).buffer(0) if keep else mp.convex_hull
        except Exception:
            geom = gdf.unary_union.convex_hull
    else:
        geom = gdf.unary_union.convex_hull

    if buffer_m and buffer_m != 0:
        geom = gpd.GeoSeries([geom], crs=work_crs).buffer(buffer_m).iloc[0]

    return gpd.GeoSeries([geom], crs=work_crs)


def _area_overlap_metrics(poly_pred_wgs84,
                          truth_poly_work_gs: gpd.GeoSeries,
                          work_crs: int | str = 27700):
    """
    Compute area precision/recall/F1 using projected areas.
    """
    if (poly_pred_wgs84 is None) or getattr(poly_pred_wgs84, "is_empty", True) or truth_poly_work_gs is None:
        return dict(area_precision=0.0, area_recall=0.0, area_f1=0.0)

    # project predicted polygon to work CRS
    pred = gpd.GeoSeries([poly_pred_wgs84], crs=4326).to_crs(work_crs).iloc[0]
    truth = truth_poly_work_gs.iloc[0]

    if pred.is_empty or truth.is_empty:
        return dict(area_precision=0.0, area_recall=0.0, area_f1=0.0)

    inter = pred.intersection(truth)
    inter_area = inter.area if not inter.is_empty else 0.0
    pred_area  = max(pred.area, 1e-12)
    true_area  = max(truth.area, 1e-12)

    ap = inter_area / pred_area
    ar = inter_area / true_area
    af1 = (2*ap*ar/(ap+ar)) if (ap+ar) > 0 else 0.0
    return dict(area_precision=ap, area_recall=ar, area_f1=af1)


def spatial_point_metrics(poly_pred_wgs84,
                          df_test: pd.DataFrame,
                          cell_id: str,
                          target_levels: list[str],
                          work_crs: int | str = 27700,
                          truth_shape: str = "hull",     # 'hull' or 'alpha'
                          truth_buffer_m: float = 0.0,
                          neg_ratio: float = 1.0,        # negatives per positive in point eval
                          rng: int = 42):
    """
    Unified evaluator: returns point- and area-based metrics + hybrid.
    - poly_pred_wgs84: shapely Polygon/MultiPolygon in EPSG:4326
    - df_test: your filtered test window (inside AOI), ALL cells for that month
    - cell_id: the cell we're evaluating
    - target_levels: which signal categories count as 'positive'
    """
    # ---------- AREA METRICS ----------
    df_true_pts = df_test[(df_test["unique_cell"] == cell_id) &
                          (df_test["signal_level_category"].isin(target_levels))]

    truth_poly_gs = _truth_polygon_from_points(
        df_true_pts, work_crs=work_crs, method=truth_shape, buffer_m=truth_buffer_m
    )
    area_metrics = _area_overlap_metrics(poly_pred_wgs84, truth_poly_gs, work_crs=work_crs)

    # ---------- POINT METRICS (balanced) ----------
    if df_test.empty:
        point_metrics = dict(point_precision=0.0, point_recall=0.0, point_f1=0.0)
    else:
        pos = df_test[df_test["unique_cell"] == cell_id]
        neg = df_test[df_test["unique_cell"] != cell_id]

        if pos.empty:
            point_metrics = dict(point_precision=0.0, point_recall=0.0, point_f1=0.0)
        else:
            n_pos = len(pos)
            n_neg = min(len(neg), int(np.ceil(neg_ratio * n_pos)))
            neg_sample = resample(neg, replace=False, n_samples=n_neg, random_state=rng) if n_neg > 0 else neg.iloc[0:0]

            eval_df = pd.concat([pos, neg_sample], ignore_index=True)

            # labels by membership (not by level — we already filtered truth area by level)
            y_true = (eval_df["unique_cell"] == cell_id).to_numpy()

            # predictions: polygon contains?
            if (poly_pred_wgs84 is None) or getattr(poly_pred_wgs84, "is_empty", True):
                y_pred = np.zeros(len(eval_df), dtype=bool)
            else:
                P = prep(poly_pred_wgs84)
                y_pred = np.array([P.contains(Point(x, y))
                                   for x, y in zip(eval_df["longitude"], eval_df["latitude"])], dtype=bool)

            pp = precision_score(y_true, y_pred, zero_division=0)
            pr = recall_score(y_true, y_pred, zero_division=0)
            pf1 = f1_score(y_true, y_pred, zero_division=0)
            point_metrics = dict(point_precision=pp, point_recall=pr, point_f1=pf1)

    # ---------- HYBRID ----------
    hybrid = 0.5 * (point_metrics["point_f1"] + area_metrics["area_f1"])

    out = dict(**point_metrics, **area_metrics, hybrid_f1=hybrid)
    return out


def build_cell_level_results(
    df: pd.DataFrame,
    cell_id: str,
    months: Sequence[str],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Build evaluation metrics for a single cell across sequential months."""

    df_cell = df[df["unique_cell"] == cell_id]

    # Build a coarse AOI hull for OSM context from train data
    full_ch = generate_convex_hull_geom(df_cell, quantile=0.99)

    builder = (
        ContextBuilder(aoi_geom=full_ch, work_crs=27700)
        .with_layer("buildings", enabled=True)
        .with_layer("roads", enabled=True, network_type="drive")
    )

    # builder.register_layer("greenspace", builder._fetch_greenspace)
    # builder.register_layer("water", builder._fetch_water)

    ctx_pack, featurizer, feature_cols = (
        builder
        # .with_layer("greenspace", enabled=True)
        # .with_layer("water", enabled=False)
        .with_rings([100, 250])
        .build()
    )
    aoi = ctx_pack["aoi_gdf"]

    results = []  # one row per month

    target_levels = [
        "1. Very Weak", "2. Weak", "3. Moderate", "4. Strong", "5. Very Strong"
    ]

    # for month in months[:-1]:
    for month in months[:-2]:

        # ---- TRAIN / TEST SPLIT (cell-specific) ----
        df_train = df_cell[
            pd.to_datetime(df_cell["timestamp"]).dt.to_period("M")
            == pd.to_datetime(month).to_period("M")]

        test_month = (pd.to_datetime(month).to_period("M") + 1).strftime("%Y-%m-%d")
        df_test_all = df_cell[
            pd.to_datetime(df_cell["timestamp"]).dt.to_period("M")
            == pd.to_datetime(test_month).to_period("M")]

        # Filter test to AOI (bbox → precise within)
        aoi_wgs = aoi.to_crs(4326)
        aoi_poly = aoi_wgs.iloc[0].geometry
        minx, miny, maxx, maxy = aoi_poly.bounds
        mask_bbox = (
            df_test_all["longitude"].between(minx, maxx) &
            df_test_all["latitude"].between(miny,  maxy))
        df_test_win = df_test_all.loc[mask_bbox].copy()
        gtest = gpd.GeoDataFrame(
            df_test_win,
            geometry=gpd.points_from_xy(
                df_test_win["longitude"], df_test_win["latitude"]),
            crs=4326
        )
        inside = gtest.within(aoi_poly)
        df_test = df_test_win.loc[inside].copy()

        if len(df_train) < 3 or len(df_test) == 0:
            print(f"[{month}] Skipping (train={len(df_train)}, test={len(df_test)})")
            continue

        print(f"\n=== Month {month} → Test {test_month} | train={len(df_train)} test={len(df_test)} ===")

        # ------------------------------------------------
        # 1) CONVEX HULL (baseline)
        # ------------------------------------------------
        try:
            poly_ch = generate_convex_hull_geom(df_train, quantile=0.98)
        except Exception as e:
            print(f"[CH] hull failed: {e}")
            poly_ch = None

        m_ch = spatial_point_metrics(
            poly_ch, df_test, cell_id, target_levels,
            work_crs=27700, truth_shape="hull",
            truth_buffer_m=0.0, neg_ratio=1.0)

        # ------------------------------------------------
        # 2) VANILLA OC-SVM on (lon,lat) only  — pick best by point F1
        # ------------------------------------------------
        svm_grid = [
            {"kernel": "rbf", "nu": 0.02, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.02, "gamma": 2.0e4},
            {"kernel": "rbf", "nu": 0.04, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.04, "gamma": 2.0e4},
            {"kernel": "rbf", "nu": 0.06, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.06, "gamma": 2.0e4},
        ]
        best_van_poly, best_van_metrics, best_van_args = None, {"point_f1": -1}, None
        for args in svm_grid:
            try:
                poly_svm_try = generate_svm_boundary_geom(df_train, **args)
            except Exception as e:
                print(f"[SVM] {args} failed: {e}")
                continue
            m_try = spatial_point_metrics(
                poly_svm_try, df_test, cell_id, target_levels,
                work_crs=27700, truth_shape="hull",
                truth_buffer_m=0.0, neg_ratio=1.0)
            if m_try["point_f1"] > best_van_metrics["point_f1"]:
                best_van_poly, best_van_metrics, best_van_args = poly_svm_try, m_try, args

        poly_svm = best_van_poly
        m_svm = best_van_metrics

        # ------------------------------------------------
        # 3) CONTEXT OC-SVM (OSM features) — small grid
        # ------------------------------------------------
        X_train_ctx, _ = featurizer(df_train[["longitude","latitude"]].to_numpy())
        scaler_ctx = StandardScaler().fit(X_train_ctx)
        Xtr_ctx = scaler_ctx.transform(X_train_ctx)

        ctx_grid = [
            # {"nu": 0.02, "resolution": 300},
            # {"nu": 0.04, "resolution": 300},
            # {"nu": 0.06, "resolution": 300},
            {"kernel": "rbf", "nu": 0.06, "gamma": 2.0e4},
            {"nu": 0.04, "resolution": 500},  # smoother contour
        ]
        best_ctx_poly, best_ctx_metrics, best_ctx_args = None, {"point_f1": -1}, None

        for cfg in ctx_grid:
            try:
                clf_ctx = svm.OneClassSVM(kernel="rbf", nu=cfg["nu"], gamma="scale").fit(Xtr_ctx)
                poly_ctx_try = generate_context_svm_boundary_geom(
                    aoi_gdf=aoi, featurizer=featurizer, scaler=scaler_ctx, clf=clf_ctx,
                    resolution=cfg["resolution"]
                )
            except Exception as e:
                print(f"[CTX] {cfg} failed: {e}")
                continue

            m_try = spatial_point_metrics(
                poly_ctx_try, df_test, cell_id, target_levels,
                work_crs=27700, truth_shape="hull",
                truth_buffer_m=0.0, neg_ratio=1.0)
            if m_try["point_f1"] > best_ctx_metrics["point_f1"]:
                best_ctx_poly, best_ctx_metrics, best_ctx_args = poly_ctx_try, m_try, cfg

        poly_ctx = best_ctx_poly
        m_ctx = best_ctx_metrics

        # ------------------------------------------------
        # One tidy row with both point- and area-based metrics + hybrid
        # ------------------------------------------------

        # CH results
        results.append({
            "flavour": "convex hull",
            "month": str(pd.to_datetime(month).date()),
            "test_month": str(pd.to_datetime(test_month).date()),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "poly": poly_ch,

            # Convex hull
            "args":            None,
            "point_precision": m_ch["point_precision"],
            "point_recall":    m_ch["point_recall"],
            "point_f1":        m_ch["point_f1"],
            "area_precision":  m_ch["area_precision"],
            "area_recall":     m_ch["area_recall"],
            "area_f1":         m_ch["area_f1"],
            "hybrid_f1":       m_ch["hybrid_f1"],
        })

        # Vanilla SVM results
        results.append({
            "flavour": "vanilla svm",
            "month": str(pd.to_datetime(month).date()),
            "test_month": str(pd.to_datetime(test_month).date()),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "poly": poly_svm,

            # Vanilla SVM
            "args":            best_van_args,
            "point_precision": m_svm["point_precision"],
            "point_recall":    m_svm["point_recall"],
            "point_f1":        m_svm["point_f1"],
            "area_precision":  m_svm["area_precision"],
            "area_recall":     m_svm["area_recall"],
            "area_f1":         m_svm["area_f1"],
            "hybrid_f1":       m_svm["hybrid_f1"],
        })

        # Context SVM
        results.append({
            "flavour": "context svm",
            "month": str(pd.to_datetime(month).date()),
            "test_month": str(pd.to_datetime(test_month).date()),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "poly": poly_ctx,

            # Context SVM
            "args":            best_ctx_args,
            "point_precision": m_ctx["point_precision"],
            "point_recall":    m_ctx["point_recall"],
            "point_f1":        m_ctx["point_f1"],
            "area_precision":  m_ctx["area_precision"],
            "area_recall":     m_ctx["area_recall"],
            "area_f1":         m_ctx["area_f1"],
            "hybrid_f1":       m_ctx["hybrid_f1"],
        })

        break

    # Final table
    metrics_df = pd.DataFrame(results)

    return metrics_df, aoi


def plot_data(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    cell_id: str,
    base_aoi: Optional[gpd.GeoDataFrame] = None,
    map_instance: Optional[KeplerGl] = None,
) -> KeplerGl:
    """Overlay training points and model polygons for a cell on a Kepler.gl map."""

    month = results_df["month"].unique()[0]
    df_val = df[df["unique_cell"] == cell_id]
    df_val = df_val[
        pd.to_datetime(df_val["timestamp"]).dt.to_period("M")
        == pd.to_datetime(month).to_period("M")
    ]

    poly_ctx = results_df[results_df["flavour"] == "context svm"]["poly"].values[0]
    poly_svm = results_df[results_df["flavour"] == "vanilla svm"]["poly"].values[0]
    poly_ch = results_df[results_df["flavour"] == "convex hull"]["poly"].values[0]

    gdf_poly = gpd.GeoDataFrame(
        {"name": ["svm_ctx", "svm", "ch"]},
        geometry=[poly_ctx, poly_svm, poly_ch],
        crs="EPSG:4326",
    )

    gdf_train = gpd.GeoDataFrame(
        df_val.copy(),
        geometry=gpd.points_from_xy(df_val.longitude, df_val.latitude),
        crs="EPSG:4326",
    )

    m = map_instance or KeplerGl(height=600)
    m.add_data(data=gdf_train, name="TrainPoints")
    m.add_data(data=gdf_poly, name="Polygons")

    centroid_geom = None
    if base_aoi is not None and not base_aoi.empty:
        centroid_geom = base_aoi.to_crs(4326).geometry.iloc[0].centroid
    else:
        for poly in (poly_ctx, poly_svm, poly_ch):
            if poly is not None and not getattr(poly, "is_empty", True):
                centroid_geom = poly.centroid
                break

    if centroid_geom is not None:
        m.config = {
            "version": "v1",
            "config": {
                "mapState": {
                    "latitude": centroid_geom.y,
                    "longitude": centroid_geom.x,
                    "zoom": 12,
                    "pitch": 0,
                    "bearing": 0,
                }
            },
        }

    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute coverage polygons for selected cells.")
    parser.add_argument(
        "--cells",
        nargs="*",
        help="Explicit list of cell IDs to process. Defaults to a curated sample.",
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        help="Optional CSV shards to load instead of the default dataset file list.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS_REQUIRED,
        help="Minimum number of samples per cell/month required to include a cell.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Optional path to write the combined metrics table as CSV.",
    )
    parser.add_argument(
        "--kepler-html",
        type=Path,
        help="Directory to dump Kepler.gl HTML exports (one per cell).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        df_raw = load_measurements(args.data_files)
    except Exception as err:
        raise RuntimeError("Failed to load measurement dataset") from err

    print(f"Loaded {len(df_raw):,} measurements across {df_raw['unique_cell'].nunique()} cells.")

    df = prepare_signal_categories(df_raw)

    months = DEFAULT_MONTHS
    available_cells = get_sufficient_data_cells(df, months, args.min_points)
    print(f"Cells with >= {args.min_points} samples per active month: {len(available_cells)}")

    requested_cells = args.cells if args.cells else DEFAULT_CELL_IDS
    if not requested_cells:
        requested_cells = available_cells

    missing_cells = [cell_id for cell_id in requested_cells if cell_id not in available_cells]
    if missing_cells:
        warning = ", ".join(missing_cells)
        print(f"Warning: Skipping {len(missing_cells)} requested cells without sufficient data: {warning}")

    target_cells = [cell_id for cell_id in requested_cells if cell_id in available_cells]
    if not target_cells:
        print("No cells met the inclusion criteria. Nothing to do.")
        return

    combined_frames: list[pd.DataFrame] = []
    map_artifacts: dict[str, KeplerGl] = {}

    for cell_id in target_cells:
        print(f"\nProcessing cell {cell_id}...")
        metrics_df, aoi = build_cell_level_results(df, cell_id, months)
        metrics_df.insert(0, "cell_id", cell_id)
        combined_frames.append(metrics_df)

        if args.kepler_html is not None:
            map_artifacts[cell_id] = plot_data(df, metrics_df, cell_id, base_aoi=aoi)

        metrics_df.to_csv(f'processed_cells/{cell_id}.csv', index=False)

    combined = pd.concat(combined_frames, ignore_index=True)
    print("\nMetrics preview:")
    print(
        combined[
            ["cell_id", "flavour", "month", "test_month", "point_f1", "area_f1", "hybrid_f1"]
        ]
    )

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.metrics_out, index=False)
        print(f"\nWrote metrics to {args.metrics_out}.")

    if args.kepler_html and map_artifacts:
        output_dir = args.kepler_html
        output_dir.mkdir(parents=True, exist_ok=True)
        for cell_id, map_obj in map_artifacts.items():
            out_file = output_dir / f"{cell_id}.html"
            map_obj.save_to_html(file_name=str(out_file))
            print(f"Saved Kepler.gl map for {cell_id} to {out_file}.")


if __name__ == "__main__":
    main()
