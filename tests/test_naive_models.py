#!/usr/bin/env python3
# tests/test_naive_models.py
"""
Unit tests for naive forecasting models.

Tests cover:
1. Model initialization and configuration
2. Fitting with various data shapes
3. Prediction generation
4. Edge cases (missing values, new items)
5. Error handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.naive_models import (
    SimpleNaiveModel,
    SeasonalNaiveModel,
    WeightedMovingAverageModel
)


class TestSimpleNaiveModel(unittest.TestCase):
    """Test suite for SimpleNaiveModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleNaiveModel(config={
            'fill_method': 'forward_fill',
            'default_value': 0.0
        })
        
        # Create simple test data
        self.dates = pd.date_range('2025-01-01', periods=100, freq='h')
        self.X_train = pd.DataFrame({
            'store_id': [1, 1, 1, 2, 2, 2] * 16 + [1, 1, 1, 2],
            'product_id': [10, 11, 12, 10, 11, 12] * 16 + [10, 11, 12, 10],
            'dt': self.dates[:100]
        })
        self.y_train = np.random.uniform(0, 10, 100)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "SimpleNaive")
        self.assertFalse(self.model.is_fitted)
        self.assertEqual(self.model.fill_method, 'forward_fill')
        self.assertEqual(self.model.default_value, 0.0)
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(self.model.is_fitted)
        self.assertGreater(len(self.model.last_values), 0)
    
    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        X_test = self.X_train.head(10)
        with self.assertRaises(ValueError):
            self.model.predict(X_test)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        self.model.fit(self.X_train, self.y_train)
        X_test = self.X_train.head(10)
        predictions = self.model.predict(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(predictions >= 0))
    
    def test_new_store_product_handling(self):
        """Test handling of new store-product combinations."""
        self.model.fit(self.X_train, self.y_train)
        
        # Create test data with new store-product
        X_test = pd.DataFrame({
            'store_id': [999, 999],
            'product_id': [999, 999],
        })
        
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.all(predictions == self.model.default_value))
    
    def test_model_summary(self):
        """Test model summary generation."""
        self.model.fit(self.X_train, self.y_train)
        summary = self.model.get_model_summary()
        
        self.assertIn('unique_items', summary)
        self.assertIn('fill_method', summary)
        self.assertIn('default_value', summary)
        self.assertGreater(summary['unique_items'], 0)


class TestSeasonalNaiveModel(unittest.TestCase):
    """Test suite for SeasonalNaiveModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SeasonalNaiveModel(config={
            'seasonal_period': 168,
            'fallback_method': 'simple_naive',
            'default_value': 0.0
        })
        
        # Create test data with 2 weeks of hourly data
        self.dates = pd.date_range('2025-01-01', periods=336, freq='h')  # 14 days
        self.X_train = pd.DataFrame({
            'store_id': [1, 2] * 168,
            'product_id': [10, 10] * 168,
            'dt': self.dates
        })
        # Create repeating pattern
        self.y_train = np.tile(np.sin(np.arange(168) * 2 * np.pi / 24), 2) * 5 + 5
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "SeasonalNaive")
        self.assertEqual(self.model.seasonal_period, 168)
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(self.model.is_fitted)
        self.assertGreater(len(self.model.seasonal_values), 0)
    
    def test_seasonal_indexing(self):
        """Test hour-of-week indexing."""
        self.model.fit(self.X_train, self.y_train)
        
        # Check that seasonal values are properly indexed
        for store_product_key, seasonal_dict in self.model.seasonal_values.items():
            # Should have values for hours 0-167
            self.assertLessEqual(max(seasonal_dict.keys()), 167)
            self.assertGreaterEqual(min(seasonal_dict.keys()), 0)
    
    def test_predict_with_seasonal(self):
        """Test prediction using seasonal patterns."""
        self.model.fit(self.X_train, self.y_train)
        
        # Create test data for same hour in future week
        test_dates = pd.date_range('2025-01-15', periods=24, freq='h')
        X_test = pd.DataFrame({
            'store_id': [1] * 24,
            'product_id': [10] * 24,
            'dt': test_dates
        })
        
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), 24)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_insufficient_data_fallback(self):
        """Test fallback when insufficient seasonal data."""
        # Create data with only 1 day (insufficient for 168-hour seasonal)
        short_dates = pd.date_range('2025-01-01', periods=24, freq='h')
        X_short = pd.DataFrame({
            'store_id': [1] * 24,
            'product_id': [10] * 24,
            'dt': short_dates
        })
        y_short = np.random.uniform(0, 10, 24)
        
        self.model.fit(X_short, y_short)
        
        # Should have fallback values
        self.assertGreater(len(self.model.fallback_values), 0)


class TestWeightedMovingAverageModel(unittest.TestCase):
    """Test suite for WeightedMovingAverageModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = WeightedMovingAverageModel(config={
            'windows': [6, 24, 168],
            'weights': 'exponential',
            'alpha': 0.7,
            'default_value': 0.0
        })
        
        # Create test data
        self.dates = pd.date_range('2025-01-01', periods=200, freq='h')
        self.X_train = pd.DataFrame({
            'store_id': [1] * 200,
            'product_id': [10] * 200,
            'dt': self.dates
        })
        # Create trending data with seasonality
        trend = np.linspace(0, 5, 200)
        seasonal = 2 * np.sin(np.arange(200) * 2 * np.pi / 24)
        self.y_train = trend + seasonal + np.random.normal(0, 0.1, 200)
        self.y_train = np.maximum(self.y_train, 0)  # Ensure non-negative
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "WeightedMovingAverage")
        self.assertEqual(self.model.windows, [6, 24, 168])
        self.assertEqual(self.model.alpha, 0.7)
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(self.model.is_fitted)
        self.assertGreater(len(self.model.historical_data), 0)
    
    def test_window_stats(self):
        """Test that window statistics are computed correctly."""
        self.model.fit(self.X_train, self.y_train)
        
        for store_product_key, window_stats in self.model.historical_data.items():
            for window in self.model.windows:
                self.assertIn(window, window_stats)
                self.assertIn('mean', window_stats[window])
                self.assertIn('std', window_stats[window])
                self.assertIn('count', window_stats[window])
    
    def test_predict(self):
        """Test prediction generation."""
        self.model.fit(self.X_train, self.y_train)
        
        test_dates = pd.date_range('2025-01-10', periods=10, freq='h')
        X_test = pd.DataFrame({
            'store_id': [1] * 10,
            'product_id': [10] * 10,
            'dt': test_dates
        })
        
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_exponential_weighting(self):
        """Test exponential weighting scheme."""
        self.model.fit(self.X_train, self.y_train)
        
        window_stats = self.model.historical_data[list(self.model.historical_data.keys())[0]]
        combined = self.model._combine_windows(window_stats)
        
        self.assertTrue(isinstance(combined, float))
        self.assertTrue(np.isfinite(combined))
    
    def test_equal_weighting(self):
        """Test equal weighting option."""
        model_equal = WeightedMovingAverageModel(config={
            'windows': [6, 24, 168],
            'weights': 'equal',
        })
        
        model_equal.fit(self.X_train, self.y_train)
        predictions_equal = model_equal.predict(self.X_train.head(10))
        
        self.assertEqual(len(predictions_equal), 10)
        self.assertTrue(np.all(np.isfinite(predictions_equal)))


class TestModelComparison(unittest.TestCase):
    """Test suite for comparing models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create realistic training data
        np.random.seed(42)
        self.dates = pd.date_range('2025-01-01', periods=500, freq='h')
        
        self.X_train = pd.DataFrame({
            'store_id': np.repeat([1, 2, 3], 500),
            'product_id': np.tile([10, 20], 750),
            'dt': pd.concat([pd.Series(self.dates)] * 3, ignore_index=True)[:1500]
        })
        
        # Create synthetic sales with seasonality
        n = len(self.X_train)
        hours = self.X_train['dt'].dt.hour.values
        days = self.X_train['dt'].dt.dayofweek.values
        
        daily_pattern = 3 * np.sin(hours * 2 * np.pi / 24)
        weekly_pattern = 1 * np.sin(days * 2 * np.pi / 7)
        noise = np.random.normal(0, 0.5, n)
        
        self.y_train = np.maximum(5 + daily_pattern + weekly_pattern + noise, 0)
    
    def test_all_models_train(self):
        """Test that all models can be trained on same data."""
        models = {
            'simple': SimpleNaiveModel(),
            'seasonal': SeasonalNaiveModel(),
            'wma': WeightedMovingAverageModel()
        }
        
        for name, model in models.items():
            try:
                model.fit(self.X_train, self.y_train)
                self.assertTrue(model.is_fitted, f"{name} not fitted")
            except Exception as e:
                self.fail(f"{name} failed to fit: {e}")
    
    def test_all_models_predict(self):
        """Test that all models can generate predictions."""
        models = {
            'simple': SimpleNaiveModel(),
            'seasonal': SeasonalNaiveModel(),
            'wma': WeightedMovingAverageModel()
        }
        
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_train.head(50))
            
            self.assertEqual(len(predictions), 50, f"{name} wrong prediction count")
            self.assertTrue(np.all(np.isfinite(predictions)), f"{name} has NaN predictions")


if __name__ == '__main__':
    unittest.main()
