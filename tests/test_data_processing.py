"""
Unit tests for src/data_processing.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_processing import (
    extract_time_features,
    compute_rfm_features,
    handle_missing_values,
    encode_categoricals,
    build_feature_matrix,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic transaction DataFrame matching the Xente schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "TransactionId": [f"T{i}" for i in range(n)],
            "BatchId": [f"B{i % 10}" for i in range(n)],
            "AccountId": [f"A{i % 20}" for i in range(n)],
            "SubscriptionId": [f"S{i % 5}" for i in range(n)],
            "CustomerId": [f"C{i % 15}" for i in range(n)],
            "CurrencyCode": ["UGX"] * n,
            "CountryCode": [256] * n,
            "ProviderId": [f"P{i % 4}" for i in range(n)],
            "ProductId": [f"Prod{i % 8}" for i in range(n)],
            "ProductCategory": np.random.choice(
                ["airtime", "financial_services", "utility_bill", "tv"], n
            ),
            "ChannelId": np.random.choice(["ChannelId_1", "ChannelId_2", "ChannelId_3"], n),
            "Amount": np.random.uniform(-5000, 50000, n),
            "Value": np.abs(np.random.uniform(0, 50000, n)),
            "TransactionStartTime": pd.date_range(
                "2019-01-01", periods=n, freq="6h"
            ),
            "PricingStrategy": np.random.randint(0, 4, n),
            "FraudResult": np.random.randint(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# extract_time_features
# ---------------------------------------------------------------------------

class TestExtractTimeFeatures:
    def test_adds_expected_columns(self, sample_df):
        result = extract_time_features(sample_df)
        for col in ["tx_hour", "tx_day_of_week", "tx_month", "tx_year"]:
            assert col in result.columns

    def test_does_not_mutate_original(self, sample_df):
        original_cols = set(sample_df.columns)
        extract_time_features(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_hour_range(self, sample_df):
        result = extract_time_features(sample_df)
        assert result["tx_hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df):
        result = extract_time_features(sample_df)
        assert result["tx_day_of_week"].between(0, 6).all()


# ---------------------------------------------------------------------------
# compute_rfm_features
# ---------------------------------------------------------------------------

class TestComputeRFMFeatures:
    def test_adds_rfm_columns(self, sample_df):
        result = compute_rfm_features(sample_df)
        for col in ["recency", "frequency", "monetary"]:
            assert col in result.columns

    def test_recency_is_non_negative(self, sample_df):
        result = compute_rfm_features(sample_df)
        assert (result["recency"] >= 0).all()

    def test_frequency_is_positive(self, sample_df):
        result = compute_rfm_features(sample_df)
        assert (result["frequency"] > 0).all()

    def test_row_count_unchanged(self, sample_df):
        result = compute_rfm_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------

class TestHandleMissingValues:
    def test_no_nulls_after_imputation(self, sample_df):
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "Amount"] = np.nan
        df_with_nulls.loc[5, "Value"] = np.nan
        result = handle_missing_values(df_with_nulls)
        assert result[["Amount", "Value"]].isnull().sum().sum() == 0

    def test_original_not_mutated(self, sample_df):
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "Amount"] = np.nan
        handle_missing_values(df_with_nulls)
        assert pd.isna(df_with_nulls.loc[0, "Amount"])


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_original_cat_cols_removed(self, sample_df):
        result = encode_categoricals(sample_df, cat_cols=["ProductCategory"])
        assert "ProductCategory" not in result.columns

    def test_one_hot_columns_created(self, sample_df):
        result = encode_categoricals(sample_df, cat_cols=["ProductCategory"])
        ohe_cols = [c for c in result.columns if c.startswith("ProductCategory_")]
        assert len(ohe_cols) > 0

    def test_unknown_cat_col_ignored(self, sample_df):
        # Should not raise even if a listed column is missing
        result = encode_categoricals(sample_df, cat_cols=["NonExistentCol"])
        assert set(result.columns) == set(sample_df.columns)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_target_excluded_from_X(self, sample_df):
        X, y = build_feature_matrix(sample_df)
        assert "FraudResult" not in X.columns

    def test_y_matches_fraud_result(self, sample_df):
        _, y = build_feature_matrix(sample_df)
        assert y is not None
        pd.testing.assert_series_equal(y.reset_index(drop=True), sample_df["FraudResult"].reset_index(drop=True))

    def test_id_columns_excluded(self, sample_df):
        X, _ = build_feature_matrix(sample_df)
        id_cols = ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"]
        for col in id_cols:
            assert col not in X.columns


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_correct_split_sizes(self, sample_df):
        X, y = build_feature_matrix(
            encode_categoricals(
                compute_rfm_features(
                    extract_time_features(sample_df)
                )
            )
        )
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05

    def test_reproducibility(self, sample_df):
        X, y = build_feature_matrix(
            encode_categoricals(
                compute_rfm_features(
                    extract_time_features(sample_df)
                )
            )
        )
        split1 = split_data(X, y, random_state=42)
        split2 = split_data(X, y, random_state=42)
        pd.testing.assert_frame_equal(split1[0], split2[0])
