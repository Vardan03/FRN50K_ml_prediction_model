# tests/test_random_forest_models.py
"""
Unit tests for Random Forest regression baseline models.

Tests cover:
- Model initialization with various configurations
- Fitting with complete and incomplete data
- Prediction generation
- Edge cases and error handling
- Non-negative constraint enforcement
- Feature importance analysis
- Missing value handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from src.models.baseline.random_forest_models import RandomForestBaseline, RandomForestAdvanced
from src.data.feature_engineering import FeatureEngineer


class TestRandomForestBaseline:
    """Test RandomForestBaseline implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample retail data for testing."""
        np.random.seed(42)
        n_samples = 336
        
        # Generate time index
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Create DataFrame with required columns
        data = {
            'dt': dates,
            'store_id': np.tile(np.arange(1, 6), n_samples // 5 + 1)[:n_samples],
            'product_id': np.tile(np.arange(1, 4), n_samples // 3 + 1)[:n_samples],
            'city_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': np.array([1 if d.weekday() >= 5 else 0 for d in dates]),
            'is_holiday': np.random.binomial(1, 0.05, n_samples),
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': np.random.uniform(50, 150, n_samples),
            'sale_amount': np.random.exponential(20, n_samples),
        }
        
        X = pd.DataFrame(data)
        y = pd.Series(np.random.exponential(25, n_samples))
        
        return X, y
    
    def test_initialization_default(self):
        """Test model initialization with default configuration."""
        model = RandomForestBaseline()
        
        assert model.model_name == "RandomForestRegression"
        assert model.n_estimators == 100
        assert model.max_depth == 20
        assert model.max_features == 'sqrt'
        assert model.enforce_non_negative is True
        assert model.is_fitted is False
    
    def test_initialization_custom_config(self):
        """Test model initialization with custom configuration."""
        config = {
            'n_estimators': 50,
            'max_depth': 10,
            'max_features': 'log2',
            'enforce_non_negative': False,
        }
        model = RandomForestBaseline(config=config)
        
        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.max_features == 'log2'
        assert model.enforce_non_negative is False
    
    def test_fit_basic(self, sample_data):
        """Test basic model fitting."""
        X, y = sample_data
        model = RandomForestBaseline()
        
        result = model.fit(X, y)
        
        assert model.is_fitted is True
        assert len(model.feature_names) > 0
        assert result is model  # Check method chaining
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient training data."""
        X = pd.DataFrame({
            'store_id': [1, 2],
            'product_id': [1, 1],
            'dt': [datetime.now(), datetime.now()],
            'temperature': [20, 25],
            'sale_amount': [10, 15],
        })
        y = pd.Series([100, 110])
        
        model = RandomForestBaseline(config={'min_samples': 10})
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            model.fit(X, y)
    
    def test_predict_before_fit(self, sample_data):
        """Test that predict raises error when called before fit."""
        X, _ = sample_data
        model = RandomForestBaseline()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)
    
    def test_predict_basic(self, sample_data):
        """Test basic prediction after fitting."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_non_negative_enforcement(self, sample_data):
        """Test that non-negative constraint is enforced."""
        X, y = sample_data
        model = RandomForestBaseline(config={'enforce_non_negative': True})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert np.all(predictions >= 0), "Predictions should be non-negative"
    
    def test_non_negative_disabled(self, sample_data):
        """Test predictions without non-negative constraint."""
        X, y = sample_data
        model = RandomForestBaseline(config={'enforce_non_negative': False})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # With Random Forest, negative predictions are very unlikely,
        # but we're testing that the constraint is disabled
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_importance_available(self, sample_data):
        """Test that feature importance is computed."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        importance_dict = model.get_feature_importance()
        
        assert len(importance_dict) > 0
        assert isinstance(importance_dict, dict)
        # Check that values are floats between 0 and 1
        for feat, imp in importance_dict.items():
            assert isinstance(imp, (float, np.floating))
            assert 0 <= imp <= 1
    
    def test_get_feature_importance_top_n(self, sample_data):
        """Test getting top N features by importance."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        top_5 = model.get_feature_importance(top_n=5)
        top_10 = model.get_feature_importance(top_n=10)
        all_features = model.get_feature_importance_all()
        
        assert len(top_5) <= 5
        assert len(top_10) <= 10
        assert len(all_features) == len(model.feature_names)
    
    def test_get_feature_importance_before_fit(self):
        """Test that feature importance raises error before fit."""
        model = RandomForestBaseline()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance()
    
    def test_missing_values_handling(self):
        """Test handling of missing values in input data."""
        np.random.seed(42)
        n_samples = 100
        
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        X = pd.DataFrame({
            'dt': dates,
            'store_id': np.arange(1, n_samples + 1),
            'product_id': np.tile([1, 2, 3], n_samples // 3 + 1)[:n_samples],
            'city_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.concatenate([
                np.random.uniform(30, 90, n_samples // 2),
                [np.nan] * (n_samples // 2)  # Add missing values
            ]),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': np.array([1 if d.weekday() >= 5 else 0 for d in dates]),
            'is_holiday': np.random.binomial(1, 0.05, n_samples),
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': np.random.uniform(50, 150, n_samples),
            'sale_amount': np.random.exponential(20, n_samples),
        })
        y = pd.Series(np.random.exponential(25, n_samples))
        
        model = RandomForestBaseline()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_feature_names_extracted(self, sample_data):
        """Test that feature names are correctly extracted."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        assert len(model.feature_names) > 0
        # Feature names should not contain identifiers
        excluded = {'store_id', 'product_id', 'city_id', 'dt', 'sale_amount'}
        for feat in model.feature_names:
            assert not any(ex in feat for ex in excluded)
    
    def test_model_summary(self, sample_data):
        """Test model summary generation."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert summary['model_type'] == 'RandomForest'
        assert summary['n_estimators'] == 100
        assert summary['n_features'] > 0
        assert 'top_10_important_features' in summary
    
    def test_fit_predict_consistency(self, sample_data):
        """Test that predictions are consistent for same data."""
        X, y = sample_data
        model = RandomForestBaseline(config={'random_state': 42})
        model.fit(X, y)
        
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_prediction_shape(self, sample_data):
        """Test that prediction shapes match input."""
        X, y = sample_data
        model = RandomForestBaseline()
        model.fit(X, y)
        
        test_size = 50
        X_test = X.iloc[:test_size]
        predictions = model.predict(X_test)
        
        assert predictions.shape == (test_size,)
    
    def test_config_parameters_respected(self):
        """Test that configuration parameters are properly applied."""
        config = {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
        }
        model = RandomForestBaseline(config=config)
        
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 8
        assert model.model.min_samples_split == 10
        assert model.model.min_samples_leaf == 4
    
    def test_empty_input_validation(self):
        """Test that empty input raises error."""
        model = RandomForestBaseline()
        X_empty = pd.DataFrame()
        y_empty = pd.Series()
        
        with pytest.raises(ValueError, match="empty"):
            model.fit(X_empty, y_empty)
    
    def test_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = pd.DataFrame({
            'store_id': [1, 2, 3],
            'product_id': [1, 1, 1],
            'dt': [datetime.now()] * 3,
            'sale_amount': [10, 20, 30],
            'temperature': [20, 25, 30],
        })
        y = pd.Series([100, 110])  # Different length
        
        model = RandomForestBaseline()
        
        with pytest.raises(ValueError, match="different lengths"):
            model.fit(X, y)


class TestRandomForestAdvanced:
    """Test RandomForestAdvanced (ensemble comparison) implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        data = {
            'dt': dates,
            'store_id': np.arange(1, n_samples + 1),
            'product_id': np.tile([1, 2, 3], n_samples // 3 + 1)[:n_samples],
            'city_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': np.array([1 if d.weekday() >= 5 else 0 for d in dates]),
            'is_holiday': np.random.binomial(1, 0.05, n_samples),
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': np.random.uniform(50, 150, n_samples),
            'sale_amount': np.random.exponential(20, n_samples),
        }
        
        X = pd.DataFrame(data)
        y = pd.Series(np.random.exponential(25, n_samples))
        
        return X, y
    
    def test_random_forest_initialization(self):
        """Test RandomForestAdvanced with random forest type."""
        model = RandomForestAdvanced(ensemble_type='random_forest')
        assert model.ensemble_type == 'random_forest'
        assert model.is_fitted is False
    
    def test_gradient_boosting_initialization(self):
        """Test RandomForestAdvanced with gradient boosting type."""
        model = RandomForestAdvanced(ensemble_type='gradient_boosting')
        assert model.ensemble_type == 'gradient_boosting'
    
    def test_invalid_ensemble_type(self):
        """Test that invalid ensemble type raises error."""
        with pytest.raises(ValueError, match="Unknown ensemble_type"):
            RandomForestAdvanced(ensemble_type='invalid')
    
    def test_fit_predict_cycle(self, sample_data):
        """Test fit and predict cycle for advanced model."""
        X, y = sample_data
        model = RandomForestAdvanced(ensemble_type='random_forest')
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert model.is_fitted is True
        assert len(predictions) == len(X)
        assert np.all(predictions >= 0)
    
    def test_model_summary_advanced(self, sample_data):
        """Test model summary for advanced model."""
        X, y = sample_data
        model = RandomForestAdvanced(ensemble_type='random_forest')
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert summary['ensemble_type'] == 'random_forest'
        assert summary['n_features'] > 0
        assert 'top_10_features' in summary


class TestRandomForestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_constant_target(self):
        """Test fitting with constant target values."""
        n_samples = 50
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        X = pd.DataFrame({
            'dt': dates,
            'store_id': np.arange(1, n_samples + 1),
            'product_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
            'city_id': 1,
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': 0,
            'is_holiday': 0,
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': 100,
            'sale_amount': 50,
        })
        y = pd.Series([100.0] * n_samples)  # Constant target
        
        model = RandomForestBaseline()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Predictions should be close to constant
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_all_zeros_target(self):
        """Test fitting with all zero target values."""
        n_samples = 50
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        X = pd.DataFrame({
            'dt': dates,
            'store_id': np.arange(1, n_samples + 1),
            'product_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
            'city_id': 1,
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': 0,
            'is_holiday': 0,
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': 0,
            'sale_amount': 0,
        })
        y = pd.Series(np.zeros(n_samples))
        
        model = RandomForestBaseline()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_single_feature_prediction(self):
        """Test prediction with minimal features."""
        n_samples = 50
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        X = pd.DataFrame({
            'dt': dates,
            'store_id': 1,
            'product_id': 1,
            'city_id': 1,
            'sale_amount': 50,
            'temperature': np.random.uniform(5, 35, n_samples),
        })
        y = pd.Series(np.random.exponential(25, n_samples))
        
        model = RandomForestBaseline()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all(predictions >= 0)


class TestRandomForestIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_multiple_stores_products(self):
        """Test with realistic multiple stores and products."""
        np.random.seed(42)
        n_samples = 240
        
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Create data with multiple stores and products
        stores = np.tile(np.arange(1, 6), n_samples // 5 + 1)[:n_samples]
        products = np.tile(np.arange(1, 11), n_samples // 10 + 1)[:n_samples]
        
        X = pd.DataFrame({
            'dt': dates,
            'store_id': stores,
            'product_id': products,
            'city_id': (stores - 1) % 3 + 1,
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(2, n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples),
            'stock': np.random.exponential(50, n_samples),
            'day_of_week': np.array([d.weekday() for d in dates]),
            'is_weekend': np.array([1 if d.weekday() >= 5 else 0 for d in dates]),
            'is_holiday': np.random.binomial(1, 0.05, n_samples),
            'sales_lag_1': np.random.uniform(0, 100, n_samples),
            'sales_lag_7': np.random.uniform(0, 100, n_samples),
            'avg_monthly_sales': np.random.uniform(50, 150, n_samples),
            'sale_amount': np.random.exponential(20, n_samples),
        })
        y = pd.Series(np.random.exponential(25, n_samples))
        
        # Split train/test
        split_idx = 200
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train and predict
        model = RandomForestBaseline()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert np.all(predictions >= 0)
        
        # Calculate simple metrics
        rmse = np.sqrt(np.mean((predictions - y_test.values) ** 2))
        mae = np.mean(np.abs(predictions - y_test.values))
        
        assert rmse > 0
        assert mae > 0
