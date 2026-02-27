# tests/test_xgboost_models.py
"""
Comprehensive test suite for XGBoost baseline models.

Tests cover:
- Model initialization and configuration
- Feature engineering and preprocessing
- Training and prediction
- Edge cases and error handling
- Performance metrics
- Missing value handling
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

try:
    from src.models.baseline.xgboost_models import XGBoostBaseline, XGBoostAdvanced
    XGBOOST_AVAILABLE = True
except (ImportError, RuntimeError):
    XGBOOST_AVAILABLE = False


# Skip all tests if XGBoost is not available
pytestmark = pytest.mark.skipif(
    not XGBOOST_AVAILABLE,
    reason="XGBoost not installed"
)


class TestXGBoostBaseline:
    """Test XGBoost Baseline model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'store_id': np.random.randint(1, 5, n_samples),
            'product_id': np.random.randint(1, 10, n_samples),
            'city_id': np.random.randint(1, 3, n_samples),
            'dt': dates,
            'sale_amount': np.random.uniform(10, 100, n_samples),
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.1, n_samples),
            'discount_rate': np.random.uniform(0, 0.5, n_samples),
            'competitor_price': np.random.uniform(5, 50, n_samples),
            'stock_level': np.random.uniform(0, 1000, n_samples),
            'promotion_active': np.random.binomial(1, 0.3, n_samples),
            'customer_count': np.random.randint(1, 100, n_samples),
            'avg_temperature': np.random.uniform(5, 35, n_samples),
            'precpt': np.random.uniform(0, 50, n_samples),
            'is_evening_rush': np.random.binomial(1, 0.3, n_samples),
            'hours_stock_status': np.random.uniform(0, 24, n_samples),
        }
        
        df = pd.DataFrame(data)
        X = df.drop('sale_amount', axis=1)
        y = pd.Series(df['sale_amount'], name='sale_amount')
        
        return X, y
    
    def test_initialization_default(self):
        """Test model initialization with default config."""
        model = XGBoostBaseline()
        assert model.model is not None
        assert model.n_estimators == 100
        assert model.max_depth == 6
        assert model.learning_rate == 0.1
        assert model.enforce_non_negative is True
    
    def test_initialization_custom(self):
        """Test model initialization with custom config."""
        config = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 2.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'enforce_non_negative': False,
        }
        model = XGBoostBaseline(config=config)
        
        assert model.n_estimators == 200
        assert model.max_depth == 8
        assert model.learning_rate == 0.05
        assert model.reg_alpha == 0.1
        assert model.reg_lambda == 2.0
        assert model.subsample == 0.7
        assert model.colsample_bytree == 0.7
        assert model.enforce_non_negative is False
    
    def test_fit_basic(self, sample_data):
        """Test basic model fitting."""
        X, y = sample_data
        model = XGBoostBaseline()
        
        model.fit(X, y)
        
        assert model.model is not None
        assert len(model.feature_names) > 0
        assert model._model_feature_count > 0
    
    def test_fit_custom_config(self, sample_data):
        """Test fitting with custom hyperparameters."""
        X, y = sample_data
        config = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.2,
            'subsample': 0.6,
        }
        model = XGBoostBaseline(config=config)
        model.fit(X, y)
        
        assert model.n_estimators == 50
        assert model.max_depth == 4
        assert model.learning_rate == 0.2
    
    def test_predict_basic(self, sample_data):
        """Test basic prediction."""
        X, y = sample_data
        model = XGBoostBaseline()
        model.fit(X, y)
        
        X_test = X.iloc[:20]
        predictions = model.predict(X_test)
        
        assert predictions.shape[0] == 20
        assert np.all(predictions >= 0)  # Should be non-negative
        assert np.all(np.isfinite(predictions))
    
    def test_predict_non_negative_enforcement(self, sample_data):
        """Test non-negative prediction enforcement."""
        X, y = sample_data
        
        config = {'enforce_non_negative': True}
        model = XGBoostBaseline(config=config)
        model.fit(X, y)
        
        predictions = model.predict(X.iloc[:20])
        assert np.all(predictions >= 0)
    
    def test_predict_without_fit(self, sample_data):
        """Test prediction without fitting raises error."""
        X, _ = sample_data
        model = XGBoostBaseline()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
    
    def test_feature_engineering(self, sample_data):
        """Test that feature engineering is applied."""
        X, y = sample_data
        model = XGBoostBaseline()
        model.fit(X, y)
        
        # Should have more features after engineering
        assert len(model.feature_names) > X.shape[1]
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        model = XGBoostBaseline()
        model.fit(X, y)
        
        importance_df = model.get_feature_importance(top_n=5)
        
        assert len(importance_df) > 0
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'cumulative_importance' in importance_df.columns
        assert len(importance_df) <= 5
    
    def test_feature_importance_all(self, sample_data):
        """Test getting all feature importance."""
        X, y = sample_data
        model = XGBoostBaseline()
        model.fit(X, y)
        
        importance_df = model.get_feature_importance_all()
        
        assert len(importance_df) > 0
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_model_summary(self, sample_data):
        """Test model summary generation."""
        X, y = sample_data
        model = XGBoostBaseline()
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'n_features' in summary
        assert 'n_estimators' in summary
        assert 'max_depth' in summary
        assert 'learning_rate' in summary
        assert 'top_features' in summary
        assert summary['n_estimators'] == 100
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'store_id': np.random.randint(1, 5, n_samples),
            'product_id': np.random.randint(1, 10, n_samples),
            'city_id': np.random.randint(1, 3, n_samples),
            'dt': dates,
            'sale_amount': np.random.uniform(10, 100, n_samples),
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.1, n_samples),
            'discount_rate': np.random.uniform(0, 0.5, n_samples),
            'competitor_price': np.random.uniform(5, 50, n_samples),
            'stock_level': np.random.uniform(0, 1000, n_samples),
            'promotion_active': np.random.binomial(1, 0.3, n_samples),
            'customer_count': np.random.randint(1, 100, n_samples),
            'avg_temperature': np.random.uniform(5, 35, n_samples),
            'precpt': np.random.uniform(0, 50, n_samples),
            'is_evening_rush': np.random.binomial(1, 0.3, n_samples),
            'hours_stock_status': np.random.uniform(0, 24, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        df.loc[10:15, 'temperature'] = np.nan
        df.loc[20:25, 'humidity'] = np.nan
        df.loc[30:35, 'stock_level'] = np.nan
        
        X = df.drop('sale_amount', axis=1)
        y = pd.Series(df['sale_amount'], name='sale_amount')
        
        model = XGBoostBaseline()
        model.fit(X, y)
        
        # Should handle NaN values during prediction
        predictions = model.predict(X.iloc[:20])
        assert np.all(np.isfinite(predictions))
    
    def test_validation_set(self, sample_data):
        """Test fitting with validation set."""
        X, y = sample_data
        X_train = X.iloc[:80]
        y_train = y.iloc[:80]
        X_val = X.iloc[80:]
        y_val = y.iloc[80:]
        
        model = XGBoostBaseline(config={'n_estimators': 50})
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        assert model.model is not None
        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)
    
    def test_insufficient_samples(self, sample_data):
        """Test error with insufficient samples."""
        X, y = sample_data
        X_small = X.iloc[:5]
        y_small = y.iloc[:5]
        
        model = XGBoostBaseline(config={'min_samples': 20})
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            model.fit(X_small, y_small)


class TestXGBoostAdvanced:
    """Test Advanced XGBoost model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'store_id': np.random.randint(1, 5, n_samples),
            'product_id': np.random.randint(1, 10, n_samples),
            'city_id': np.random.randint(1, 3, n_samples),
            'dt': dates,
            'sale_amount': np.random.uniform(10, 100, n_samples),
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.1, n_samples),
            'discount_rate': np.random.uniform(0, 0.5, n_samples),
            'competitor_price': np.random.uniform(5, 50, n_samples),
            'stock_level': np.random.uniform(0, 1000, n_samples),
            'promotion_active': np.random.binomial(1, 0.3, n_samples),
            'customer_count': np.random.randint(1, 100, n_samples),
            'avg_temperature': np.random.uniform(5, 35, n_samples),
            'precpt': np.random.uniform(0, 50, n_samples),
            'is_evening_rush': np.random.binomial(1, 0.3, n_samples),
            'hours_stock_status': np.random.uniform(0, 24, n_samples),
        }
        
        df = pd.DataFrame(data)
        X = df.drop('sale_amount', axis=1)
        y = pd.Series(df['sale_amount'], name='sale_amount')
        
        return X, y
    
    def test_initialization(self):
        """Test Advanced model initialization."""
        model = XGBoostAdvanced(algorithm='xgboost')
        assert model.model is not None
    
    def test_unsupported_algorithm(self):
        """Test that unsupported algorithms raise error."""
        with pytest.raises(ValueError, match="not supported"):
            XGBoostAdvanced(algorithm='unsupported')
    
    def test_fit_predict_cycle(self, sample_data):
        """Test complete fit-predict cycle."""
        X, y = sample_data
        model = XGBoostAdvanced(algorithm='xgboost')
        
        model.fit(X, y)
        predictions = model.predict(X.iloc[:20])
        
        assert predictions.shape[0] == 20
        assert np.all(np.isfinite(predictions))
    
    def test_model_summary(self, sample_data):
        """Test model summary from Advanced wrapper."""
        X, y = sample_data
        model = XGBoostAdvanced(algorithm='xgboost')
        model.fit(X, y)
        
        summary = model.get_model_summary()
        assert 'model_name' in summary
        assert 'n_features' in summary


class TestXGBoostEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_target(self):
        """Test handling of constant target variable."""
        np.random.seed(42)
        n_samples = 50
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'store_id': np.ones(n_samples),
            'product_id': np.ones(n_samples),
            'city_id': np.ones(n_samples),
            'dt': dates,
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.zeros(n_samples),
            'discount_rate': np.zeros(n_samples),
            'competitor_price': np.ones(n_samples) * 25,
            'stock_level': np.ones(n_samples) * 500,
            'promotion_active': np.zeros(n_samples),
            'customer_count': np.ones(n_samples) * 50,
            'avg_temperature': np.ones(n_samples) * 20,
            'precpt': np.zeros(n_samples),
            'is_evening_rush': np.zeros(n_samples),
            'hours_stock_status': np.ones(n_samples) * 12,
        }
        
        df = pd.DataFrame(data)
        X = df
        y = pd.Series(np.ones(n_samples) * 50, name='sale_amount')
        
        model = XGBoostBaseline()
        model.fit(X, y)
        
        predictions = model.predict(X.iloc[:10])
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))
    
    def test_zero_values(self):
        """Test handling of zero values in target."""
        np.random.seed(42)
        n_samples = 50
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'store_id': np.random.randint(1, 3, n_samples),
            'product_id': np.random.randint(1, 5, n_samples),
            'city_id': np.ones(n_samples),
            'dt': dates,
            'temperature': np.random.uniform(5, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.zeros(n_samples),
            'discount_rate': np.random.uniform(0, 0.5, n_samples),
            'competitor_price': np.random.uniform(5, 50, n_samples),
            'stock_level': np.random.uniform(0, 100, n_samples),
            'promotion_active': np.zeros(n_samples),
            'customer_count': np.random.randint(1, 50, n_samples),
            'avg_temperature': np.random.uniform(5, 35, n_samples),
            'precpt': np.random.uniform(0, 20, n_samples),
            'is_evening_rush': np.zeros(n_samples),
            'hours_stock_status': np.random.uniform(0, 12, n_samples),
        }
        
        df = pd.DataFrame(data)
        X = df
        y = pd.Series(np.concatenate([np.zeros(25), np.random.uniform(10, 100, 25)]), 
                      name='sale_amount')
        
        model = XGBoostBaseline()
        model.fit(X, y)
        
        predictions = model.predict(X.iloc[:10])
        assert np.all(predictions >= 0)


class TestXGBoostIntegration:
    """Integration tests for complete workflows."""
    
    def test_multistore_scenario(self):
        """Test model with multiple stores and products."""
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        stores = np.random.choice([1, 2, 3, 4, 5], n_samples)
        products = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples)
        
        data = {
            'store_id': stores,
            'product_id': products,
            'city_id': np.random.choice([1, 2], n_samples),
            'dt': dates,
            'temperature': np.random.normal(20, 8, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'day_of_week': dates.dayofweek,
            'hour': dates.hour,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.05, n_samples),
            'discount_rate': np.random.uniform(0, 0.5, n_samples),
            'competitor_price': np.random.uniform(5, 50, n_samples),
            'stock_level': np.random.uniform(0, 1000, n_samples),
            'promotion_active': np.random.binomial(1, 0.2, n_samples),
            'customer_count': np.random.randint(1, 100, n_samples),
            'avg_temperature': np.random.normal(20, 8, n_samples),
            'precpt': np.random.exponential(5, n_samples),
            'is_evening_rush': np.random.binomial(1, 0.4, n_samples),
            'hours_stock_status': np.random.uniform(0, 24, n_samples),
        }
        
        df = pd.DataFrame(data)
        X = df
        y = pd.Series(
            50 + stores * 5 + products * 2 + np.random.normal(0, 10, n_samples),
            name='sale_amount'
        )
        
        model = XGBoostBaseline()
        model.fit(X, y)
        
        # Train and test split
        predictions_train = model.predict(X.iloc[:160])
        predictions_test = model.predict(X.iloc[160:])
        
        assert len(predictions_train) == 160
        assert len(predictions_test) == 40
        assert np.all(np.isfinite(predictions_train))
        assert np.all(np.isfinite(predictions_test))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
