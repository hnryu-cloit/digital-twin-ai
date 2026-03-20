"""
feature_engineering.py - Logic for transforming synthetic customer data into feature vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# Features to use for clustering
FEATURE_COLS = [
    # Demographics
    "usr_age",
    # Purchase Value
    "retention_score", "ltv_r", "val_p", "pchs_cnt", "premium_cnt",
    # Behavioral Patterns
    "cum_repchs_flg", "samsunghealth_flag", "samsungwallet_flag", "smartthings_flag",
    # App Usage
    "app_game_ratio", "app_social_ratio", "app_samsung_ratio", "avg_daily_usage_min",
]


class FeatureEngineer:
    """Class for feature engineering and normalization."""

    def __init__(self):
        self.scaler = StandardScaler()

    def process_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables and normalize features."""
        print("Performing feature engineering...")
        df = df.copy()
        
        # Categorical encoding
        df = self._encode_categorical(df)

        extra_cols = ["usr_gndr_enc", "is_active"] + [c for c in df.columns if c.startswith("region_")]
        all_feature_cols = FEATURE_COLS + extra_cols

        # Handle missing values
        df[all_feature_cols] = df[all_feature_cols].fillna(df[all_feature_cols].median())

        # Normalization
        df[all_feature_cols] = self.scaler.fit_transform(df[all_feature_cols])

        return df, all_feature_cols

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode gender and activeness, and one-hot encode regions."""
        df["usr_gndr_enc"] = (df["usr_gndr"] == "M").astype(int)
        df["is_active"] = (df["sa_activeness"].str.startswith("Active")).astype(int)

        # Region One-Hot Encoding
        if "usr_cnty_ap2" in df.columns:
            region_dummies = pd.get_dummies(df["usr_cnty_ap2"], prefix="region")
            df = pd.concat([df, region_dummies], axis=1)

        return df
