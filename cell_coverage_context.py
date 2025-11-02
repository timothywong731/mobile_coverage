from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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




def build_all_cells() -> pd.DataFrame:

    # Iterate through every month from START_DATE to END_DATE
    all_months = list(rrule.rrule(rrule.MONTHLY, dtstart=START_DATE, until=END_DATE))

    # Format as ISO dates
    all_months = [x.strftime("%Y-%m-%d") for x in all_months]

    # Load dataset from Hugging Face

    # ds = load_dataset("joefee/cell-service-data")

    # first 100k rows of the default split (often "train")
    # ds = load_dataset("joefee/cell-service-data", split="train[:100000]")

    ds = load_dataset(
        "joefee/cell-service-data",
        data_files={
            "train": [
                "np_extract_part_1.csv", "np_extract_part_2.csv",
                "np_extract_part_3.csv", "np_extract_part_4.csv",
                "np_extract_part_5.csv", "np_extract_part_6.csv",
                # "np_extract_part_7.csv", "np_extract_part_8.csv",
                # "np_extract_part_9.csv",
            ]
        },
    )

    # Convert to pandas dataframe
    df = ds['train'].to_pandas()

    print(f"len of df: {len(df)}")

    # Bin this into signal level categories
    # UK Ofcom Reference URL: 
    # https://www.ofcom.org.uk/siteassets/resources/documents/phones-telecoms-and-internet/comparing-service-quality/2025/map-your-mobile-2025-threshold-methodology.pdf
    df["signal_level_category"] = pd.cut(
        df["signal_level"],
        bins=[-np.inf, -105, -95, -82, -74, np.inf],
        labels=["1. Very Weak", "2. Weak", "3. Moderate", "4. Strong", "5. Very Strong"]
    )

    # Identify cells with sufficient data points
    # Cells must also have signal_level_category value
    min_points_required = 1

    sufficient_data_cells = duckdb.query(f"""
    WITH monthly_count AS (
        SELECT
            unique_cell, 
            date_trunc('month', CAST(timestamp AS timestamp)) as month, 
            COUNT(*) as count
        FROM df
        WHERE signal_level IS NOT NULL
        GROUP BY unique_cell, month HAVING COUNT(*) >= {min_points_required}
    )
    SELECT unique_cell 
    FROM monthly_count 
    GROUP BY unique_cell
    --HAVING COUNT(DISTINCT month) = {len(all_months)}
    """).to_df()['unique_cell'].tolist()

    return df




def main():


    df = build_all_cells()

