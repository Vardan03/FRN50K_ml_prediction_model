#!/usr/bin/env python3
"""
Unit tests for Linear Regression Baseline models.

Tests cover:
- Model initialization
- Feature engineering integration
- Training and prediction
- Edge cases (missing values, sparse data)
- Non-negative prediction enforcement
- Feature scaling
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest
import logging

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.linear_models import LinearRegressionBaseline, LinearForecastingModel
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class TestLinearRegressionBaseline(unittest.TestCase):
    """Test cases for LinearRegressionBaseline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
        cls.random_state = 42
        np.random.seed(cls.random_state)
    
    def setUp(self):
        """Create test data for each test."""
        # Create synthetic training data
        np.random.seed(self.random_state)
        dates = pd.date_range('2025-01-01', periods=200, freq='h')
        
        self.X_train = pd.DataFrame({
            'store_id': [1, 2, 3] * 67,
            'product_id': [10, 11, 12] * 67,
            'dt': list(dates)[:200],
            'city_id': [1, 1, 2] * 67,
            'discount': np.random.uniform(0, 0.3, 200),
            'holiday_flag': np.random.randint(0, 2, 200),
            'activity_flag': np.random.randint(0, 2, 200),
            'precpt': np.random.uniform(0, 10, 200),
            'avg_temperature': np.random.uniform(15, 30, 200),
            'avg_humidity': np.random.uniform(30, 80, 200),
            'avg_wind_level': np.random.uniform(0, 20, 200),
            'hours_sale': np.random.randint(0, 24, 200),
            'hours_stock_status': np.random.randint(0, 24, 200),
            'stock_hour6_22_cnt': np.random.randint(0, 20, 200),
        })
        
        # Create target variable with some trend and seasonality
        self.y_train = pd.Series(
            5 + np.linspace(0, 2, 200) + 2*np.sin(np.arange(200)*np.pi/24) + 
            np.random.normal(0, 0.5, 200)
        )
        self.y_train = np.maximum(self.y_train, 0)  # Ensure non-negative
        
        # Create test data
        self.X_test = self.X_train.copy().head(50)
        self.y_test = self.y_train.head(50)
    
    def test_initialization(self):
        """Test model initialization with different configurations."""
        # Default initialization
        model = LinearRegressionBaseline()
        self.assertIsNotNone(model)
        self.assertEqual(model.alpha, 1.0)
        self.assertTrue(model.use_ridge)
        self.assertFalse(model.is_fitted)
        
        # Custom configuration
        config = {
            'alpha': 0.5,
            'use_ridge': False,
            'enforce_non_negative': True
        }
        model2 = LinearRegressionBaseline(config=config)
        self.assertEqual(model2.alpha, 0.5)
        self.assertFalse(model2.use_ridge)
    
    def test_fit_basic(self):
        """Test basic model fitting."""
        model = LinearRegressionBaseline()
        model.fit(self.X_train, self.y_train)
        
        # Check model state after fitting
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.feature_names), 0)
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        model = LinearRegressionBaseline(config={'min_samples': 100})
        
        # Create small dataset
        X_small = self.X_train.head(10)
        y_small = self.y_train.head(10)
        
        with self.assertRaises(ValueError):
            model.fit(X_small, y_small)
    
    def test_predict_before_fit(self):
        """Test that predict raises error if model not fitted."""
        model = LinearRegressionBaseline()
        
        with self.assertRaises(ValueError):
            model.predict(self.X_test)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        model = LinearRegressionBaseline()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        # Check prediction shape and values
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(predictions >= 0))  # Non-negative constraint
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_non_negative_enforcement(self):
        """Test that negative predictions are clipped to 0."""
        model = LinearRegressionBaseline(config={'enforce_non_negative': True})
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertTrue(np.all(predictions >= 0))
    
    def test_feature_scaling(self):
        """Test that features are scaled properly."""
        model = LinearRegressionBaseline(config={'scaler_type': 'standard'})
        model.fit(self.X_train, self.y_train)
        
        # Check that scaler is fitted
        self.assertIsNotNone(model.scaler)
        
        # Check predictions are reasonable
        predictions = model.predict(self.X_test)
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertGreater(predictions.mean(), 0)
    
    def test_feature_engineering_integration(self):
        """Test that feature engineering is properly integrated."""
        model = LinearRegressionBaseline()
        model.fit(self.X_train, self.y_train)
        
        # Check that temporal features were created
        feature_types = {
            'temporal': sum(1 for f in model.feature_names if 'hour' in f or 'day' in f),
            'lag': sum(1 for f in model.feature_names if 'lag' in f),
            'rolling': sum(1 for f in model.feature_names if 'rolling' in f),
        }
        
        self.assertGreater(feature_types['temporal'], 0)
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        X_with_nan = self.X_train.copy()
        X_with_nan.loc[0:5, 'discount'] = np.nan
        X_with_nan.loc[10:15, 'avg_temperature'] = np.nan
        
        model = LinearRegressionBaseline()
        model.fit(X_with_nan, self.y_train)
        
        # Should fit successfully despite missing values
        self.assertTrue(model.is_fitted)
        
        # Should handle NaN in predictions
        predictions = model.predict(X_with_nan.head(20))
        self.assertEqual(len(predictions), 20)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_consistency_across_runs(self):
        """Test that model produces consistent predictions with same random state."""
        config = {'alpha': 1.0}
        
        model1 = LinearRegressionBaseline(config=config)
        model1.fit(self.X_train.copy(), self.y_train.copy())
        pred1 = model1.predict(self.X_test.copy())
        
        model2 = LinearRegressionBaseline(config=config)
        model2.fit(self.X_train.copy(), self.y_train.copy())
        pred2 = model2.predict(self.X_test.copy())
        
        # Predictions should be nearly identical (allowing for small floating-point differences)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
    
    def test_different_alpha_values(self):
        """Test that different alpha values produce different models."""
        model_low = LinearRegressionBaseline(config={'alpha': 0.1})
        model_low.fit(self.X_train, self.y_train)
        pred_low = model_low.predict(self.X_test)
        
        model_high = LinearRegressionBaseline(config={'alpha': 10.0})
        model_high.fit(self.X_train, self.y_train)
        pred_high = model_high.predict(self.X_test)
        
        # Predictions should be different with different regularization
        self.assertFalse(np.allclose(pred_low, pred_high))
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = LinearRegressionBaseline()
        
        # Before fitting
        summary_unfitted = model.get_model_summary()
        self.assertIn('model_name', summary_unfitted)
        self.assertFalse(summary_unfitted.get('is_fitted', False))
        
        # After fitting
        model.fit(self.X_train, self.y_train)
        summary_fitted = model.get_model_summary()
        
        self.assertTrue(summary_fitted['is_fitted'])
        self.assertIn('model_type', summary_fitted)
        self.assertIn('alpha', summary_fitted)
        self.assertIn('n_features', summary_fitted)
        self.assertGreater(summary_fitted['n_features'], 0)
    
    def test_ridge_vs_linear(self):
        """Test difference between Ridge and Linear regression."""
        model_ridge = LinearRegressionBaseline(config={'use_ridge': True, 'alpha': 1.0})
        model_ridge.fit(self.X_train, self.y_train)
        
        model_linear = LinearRegressionBaseline(config={'use_ridge': False})
        model_linear.fit(self.X_train, self.y_train)
        
        pred_ridge = model_ridge.predict(self.X_test)
        pred_linear = model_linear.predict(self.X_test)
        
        # Both should work and produce non-negative predictions
        self.assertTrue(np.all(pred_ridge >= 0))
        self.assertTrue(np.all(pred_linear >= 0))


class TestLinearForecastingModel(unittest.TestCase):
    """Test cases for legacy LinearForecastingModel."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='h')
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        
        self.y_train = pd.Series(np.random.uniform(0, 10, 100))
        self.X_test = self.X_train.head(20)
    
    def test_initialization(self):
        """Test LinearForecastingModel initialization."""
        model = LinearForecastingModel(model_type='linear')
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'linear')
    
    def test_ridge_model(self):
        """Test LinearForecastingModel with Ridge regression."""
        model = LinearForecastingModel(model_type='ridge', config={'alpha': 0.5})
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(predictions >= 0))
    
    def test_coefficients(self):
        """Test getting model coefficients."""
        model = LinearForecastingModel(model_type='linear')
        model.fit(self.X_train, self.y_train)
        
        coeffs = model.get_coefficients()
        self.assertIn('intercept', coeffs)
        self.assertEqual(len(coeffs), len(self.X_train.columns) + 1)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering with linear models."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='h')
        
        self.X = pd.DataFrame({
            'store_id': [1] * 100,
            'product_id': [10] * 100,
            'city_id': [1] * 100,
            'dt': dates,
            'discount': np.random.uniform(0, 0.3, 100),
            'holiday_flag': np.random.randint(0, 2, 100),
            'activity_flag': np.random.randint(0, 2, 100),
            'precpt': np.random.uniform(0, 10, 100),
            'avg_temperature': np.random.uniform(15, 30, 100),
            'avg_humidity': np.random.uniform(30, 80, 100),
            'avg_wind_level': np.random.uniform(0, 20, 100),
            'hours_sale': np.random.randint(0, 24, 100),
            'hours_stock_status': np.random.randint(0, 24, 100),
            'stock_hour6_22_cnt': np.random.randint(0, 20, 100),
        })
        
        self.y = pd.Series(np.random.uniform(0, 10, 100))
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline."""
        config = {
            'lag_periods': [1, 7],
            'rolling_windows': [6, 24],
            'seasonal_periods': [24],
        }
        engineer = FeatureEngineer(config=config)
        
        X_engineered = engineer.engineer_all_features(self.X)
        
        # Check that features were created
        self.assertGreater(X_engineered.shape[1], self.X.shape[1])
        
        # Check that no NaN values remain
        self.assertEqual(X_engineered.isnull().sum().sum(), 0)
    
    def test_temporal_features(self):
        """Test temporal feature creation."""
        engineer = FeatureEngineer(config={})
        X_temporal = engineer.create_temporal_features(self.X)
        
        # Check for cyclical encoding
        self.assertIn('hour_sin', X_temporal.columns)
        self.assertIn('hour_cos', X_temporal.columns)
        self.assertIn('day_of_week_sin', X_temporal.columns)
        
        # Check values are in expected range
        self.assertTrue(X_temporal['hour_sin'].between(-1, 1).all())
        self.assertTrue(X_temporal['hour_cos'].between(-1, 1).all())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Create minimal test data."""
        np.random.seed(42)
        
        self.X_minimal = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [4, 5, 6],
        })
        
        self.y_minimal = pd.Series([1, 2, 3])
    
    def test_very_small_dataset(self):
        """Test with minimal dataset."""
        model = LinearRegressionBaseline(config={'min_samples': 2})
        model.fit(self.X_minimal, self.y_minimal)
        predictions = model.predict(self.X_minimal)
        
        self.assertEqual(len(predictions), 3)
    
    def test_all_same_values(self):
        """Test with constant target variable."""
        X = pd.DataFrame({
            'f1': [1, 1, 1, 1, 1],
            'f2': [2, 2, 2, 2, 2],
        })
        y = pd.Series([5, 5, 5, 5, 5])
        
        model = LinearRegressionBaseline()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Should predict close to constant value
        self.assertTrue(np.allclose(predictions, 5, rtol=0.1))


if __name__ == '__main__':
    unittest.main()
