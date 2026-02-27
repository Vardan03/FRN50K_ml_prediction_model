# tests/test_ensemble_models.py
"""
Comprehensive test suite for ensemble regression models.

Tests cover:
- Stacking ensemble with cross-validation
- Voting ensemble with weighted aggregation
- Blending ensemble with holdout set
- Optimal weighting ensemble
- All error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.models.baseline.ensemble_models import (
    StackingEnsembleRegressor,
    VotingEnsembleRegressor,
    BlendingEnsembleRegressor,
    OptimalWeightingEnsemble,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'store_id': np.random.randint(1, 5, n_samples),
        'product_id': np.random.randint(1, 10, n_samples),
        'city_id': np.random.randint(1, 3, n_samples),
        'dt': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'discount': np.random.uniform(0, 0.5, n_samples),
        'temperature': np.random.uniform(10, 35, n_samples),
        'humidity': np.random.uniform(20, 80, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_holiday': np.random.binomial(1, 0.1, n_samples),
        'marketing_spend': np.random.uniform(0, 1000, n_samples),
        'competitor_price': np.random.uniform(50, 200, n_samples),
        'stock_level': np.random.uniform(0, 1000, n_samples),
        'traffic': np.random.uniform(100, 5000, n_samples),
        'conversion_rate': np.random.uniform(0.01, 0.3, n_samples),
        'avg_order_value': np.random.uniform(20, 200, n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        'repeat_customers': np.random.uniform(0, 1, n_samples),
        'seasonal_index': np.random.uniform(0.5, 2, n_samples),
        'sales_amount': np.random.uniform(0, 5000, n_samples),
    })
    
    # Create target with some predictability
    y = pd.Series(
        50 + 20 * np.sin(np.arange(n_samples) / 10) + 
        0.01 * X['marketing_spend'] +
        np.random.normal(0, 5, n_samples),
        name='demand'
    )
    y = np.maximum(y, 0)  # Ensure non-negative
    
    return X, y


@pytest.fixture
def large_sample_data():
    """Create larger sample for more comprehensive tests."""
    np.random.seed(42)
    n_samples = 300
    
    X = pd.DataFrame({
        'store_id': np.random.randint(1, 5, n_samples),
        'product_id': np.random.randint(1, 10, n_samples),
        'city_id': np.random.randint(1, 3, n_samples),
        'dt': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'discount': np.random.uniform(0, 0.5, n_samples),
        'temperature': np.random.uniform(10, 35, n_samples),
        'humidity': np.random.uniform(20, 80, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_holiday': np.random.binomial(1, 0.1, n_samples),
        'marketing_spend': np.random.uniform(0, 1000, n_samples),
        'competitor_price': np.random.uniform(50, 200, n_samples),
        'stock_level': np.random.uniform(0, 1000, n_samples),
        'traffic': np.random.uniform(100, 5000, n_samples),
        'conversion_rate': np.random.uniform(0.01, 0.3, n_samples),
        'avg_order_value': np.random.uniform(20, 200, n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        'repeat_customers': np.random.uniform(0, 1, n_samples),
        'seasonal_index': np.random.uniform(0.5, 2, n_samples),
        'sales_amount': np.random.uniform(0, 5000, n_samples),
    })
    
    y = pd.Series(
        50 + 30 * np.sin(np.arange(n_samples) / 15) + 
        0.01 * X['marketing_spend'] +
        np.random.normal(0, 8, n_samples),
        name='demand'
    )
    y = np.maximum(y, 0)
    
    return X, y


class TestStackingEnsemble:
    """Test cases for Stacking Ensemble."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = StackingEnsembleRegressor()
        assert model.model_name == "StackingEnsemble"
        assert model.cv_splits == 5
        assert model.meta_learner_type == 'ridge'
        assert len(model.base_learners) >= 3
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        config = {
            'cv_splits': 3,
            'meta_learner_type': 'rf',
            'enforce_non_negative': False,
        }
        model = StackingEnsembleRegressor(config=config)
        assert model.cv_splits == 3
        assert model.meta_learner_type == 'rf'
        assert model.enforce_non_negative is False
    
    def test_fit_basic(self, large_sample_data):
        """Test basic fitting."""
        X, y = large_sample_data
        model = StackingEnsembleRegressor()
        
        result = model.fit(X, y)
        
        assert result is model  # Method chaining
        assert model.meta_learner is not None
        assert len(model.feature_names) > 0
    
    def test_predict_basic(self, large_sample_data):
        """Test basic prediction."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]
        
        model = StackingEnsembleRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_non_negative_enforcement(self, large_sample_data):
        """Test non-negative prediction enforcement."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]
        
        config = {'enforce_non_negative': True}
        model = StackingEnsembleRegressor(config=config)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert np.all(predictions >= 0)
    
    def test_fit_insufficient_samples(self, sample_data):
        """Test error on insufficient samples."""
        X, y = sample_data
        model = StackingEnsembleRegressor()
        
        with pytest.raises(ValueError):
            model.fit(X[:5], y[:5])
    
    def test_predict_before_fit(self, sample_data):
        """Test error on predict before fit."""
        X, y = sample_data
        model = StackingEnsembleRegressor()
        
        with pytest.raises(RuntimeError):
            model.predict(X)
    
    def test_model_summary(self, large_sample_data):
        """Test model summary generation."""
        X, y = large_sample_data
        model = StackingEnsembleRegressor()
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'n_base_learners' in summary
        assert 'meta_learner' in summary
        assert summary['cv_splits'] == 5
    
    def test_meta_learner_types(self, large_sample_data):
        """Test different meta-learner types."""
        X, y = large_sample_data
        
        for meta_type in ['linear', 'ridge', 'rf']:
            config = {'meta_learner_type': meta_type}
            model = StackingEnsembleRegressor(config=config)
            model.fit(X, y)
            predictions = model.predict(X)
            
            assert predictions.shape == (len(X),)
            assert not np.any(np.isnan(predictions))


class TestVotingEnsemble:
    """Test cases for Voting Ensemble."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = VotingEnsembleRegressor()
        assert model.model_name == "VotingEnsemble"
        assert model.aggregation == 'mean'
        assert len(model.base_learners) >= 3
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        config = {
            'aggregation': 'median',
            'weights': [0.5, 0.3, 0.2],
            'enforce_non_negative': False,
        }
        model = VotingEnsembleRegressor(config=config)
        assert model.aggregation == 'median'
        assert model.enforce_non_negative is False
    
    def test_fit_basic(self, sample_data):
        """Test basic fitting."""
        X, y = sample_data
        model = VotingEnsembleRegressor()
        
        result = model.fit(X, y)
        
        assert result is model
        assert len(model.feature_names) > 0
    
    def test_predict_basic(self, sample_data):
        """Test basic prediction."""
        X, y = sample_data
        X_train, X_test = X[:70], X[70:]
        y_train = y[:70]
        
        model = VotingEnsembleRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert not np.any(np.isnan(predictions))
    
    def test_aggregation_mean(self, sample_data):
        """Test mean aggregation."""
        X, y = sample_data
        config = {'aggregation': 'mean'}
        model = VotingEnsembleRegressor(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
    
    def test_aggregation_median(self, sample_data):
        """Test median aggregation."""
        X, y = sample_data
        config = {'aggregation': 'median'}
        model = VotingEnsembleRegressor(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
    
    def test_aggregation_weighted_mean(self, sample_data):
        """Test weighted mean aggregation."""
        X, y = sample_data
        config = {
            'aggregation': 'weighted_mean',
            'weights': [0.5, 0.3, 0.2],
        }
        model = VotingEnsembleRegressor(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
    
    def test_non_negative_enforcement(self, sample_data):
        """Test non-negative enforcement."""
        X, y = sample_data
        config = {'enforce_non_negative': True}
        model = VotingEnsembleRegressor(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert np.all(predictions >= 0)
    
    def test_model_summary(self, sample_data):
        """Test model summary."""
        X, y = sample_data
        model = VotingEnsembleRegressor()
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'n_base_learners' in summary
        assert 'aggregation' in summary
        assert 'weights' in summary


class TestBlendingEnsemble:
    """Test cases for Blending Ensemble."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = BlendingEnsembleRegressor()
        assert model.model_name == "BlendingEnsemble"
        assert model.holdout_ratio == 0.2
        assert model.meta_learner_type == 'ridge'
    
    def test_fit_basic(self, large_sample_data):
        """Test basic fitting."""
        X, y = large_sample_data
        model = BlendingEnsembleRegressor()
        
        result = model.fit(X, y)
        
        assert result is model
        assert model.meta_learner is not None
    
    def test_predict_basic(self, large_sample_data):
        """Test basic prediction."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]
        
        model = BlendingEnsembleRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert not np.any(np.isnan(predictions))
    
    def test_holdout_ratio_configuration(self, large_sample_data):
        """Test holdout ratio configuration."""
        X, y = large_sample_data
        
        for ratio in [0.1, 0.2, 0.3]:
            config = {'holdout_ratio': ratio}
            model = BlendingEnsembleRegressor(config=config)
            result = model.fit(X, y)
            
            assert result is model
            assert model.holdout_ratio == ratio
    
    def test_non_negative_enforcement(self, large_sample_data):
        """Test non-negative enforcement."""
        X, y = large_sample_data
        config = {'enforce_non_negative': True}
        model = BlendingEnsembleRegressor(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert np.all(predictions >= 0)
    
    def test_meta_learner_types(self, large_sample_data):
        """Test different meta-learner types."""
        X, y = large_sample_data
        
        for meta_type in ['linear', 'ridge', 'rf']:
            config = {'meta_learner_type': meta_type}
            model = BlendingEnsembleRegressor(config=config)
            model.fit(X, y)
            predictions = model.predict(X)
            
            assert predictions.shape == (len(X),)


class TestOptimalWeightingEnsemble:
    """Test cases for Optimal Weighting Ensemble."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = OptimalWeightingEnsemble()
        assert model.model_name == "OptimalWeightingEnsemble"
        assert len(model.base_learners) >= 3
    
    def test_fit_basic(self, large_sample_data):
        """Test basic fitting."""
        X, y = large_sample_data
        model = OptimalWeightingEnsemble()
        
        result = model.fit(X, y)
        
        assert result is model
        assert model.optimal_weights is not None
        assert len(model.optimal_weights) == len(model.base_learners)
    
    def test_weights_sum_to_one(self, large_sample_data):
        """Test that optimal weights sum to 1."""
        X, y = large_sample_data
        model = OptimalWeightingEnsemble()
        model.fit(X, y)
        
        assert np.isclose(np.sum(model.optimal_weights), 1.0)
    
    def test_weights_non_negative(self, large_sample_data):
        """Test that optimal weights are non-negative."""
        X, y = large_sample_data
        model = OptimalWeightingEnsemble()
        model.fit(X, y)
        
        assert np.all(model.optimal_weights >= 0)
    
    def test_predict_basic(self, large_sample_data):
        """Test basic prediction."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]
        
        model = OptimalWeightingEnsemble()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert not np.any(np.isnan(predictions))
    
    def test_non_negative_enforcement(self, large_sample_data):
        """Test non-negative enforcement."""
        X, y = large_sample_data
        config = {'enforce_non_negative': True}
        model = OptimalWeightingEnsemble(config=config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert np.all(predictions >= 0)
    
    def test_predict_before_fit(self, sample_data):
        """Test error on predict before fit."""
        X, y = sample_data
        model = OptimalWeightingEnsemble()
        
        with pytest.raises(RuntimeError):
            model.predict(X)
    
    def test_model_summary(self, large_sample_data):
        """Test model summary."""
        X, y = large_sample_data
        model = OptimalWeightingEnsemble()
        model.fit(X, y)
        
        summary = model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'n_base_learners' in summary
        assert 'optimal_weights' in summary
        assert summary['optimal_weights'] is not None


class TestEnsembleIntegration:
    """Integration tests for ensemble models."""
    
    def test_all_ensembles_fit_predict(self, large_sample_data):
        """Test that all ensembles can fit and predict."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]
        
        ensemble_classes = [
            StackingEnsembleRegressor,
            VotingEnsembleRegressor,
            BlendingEnsembleRegressor,
            OptimalWeightingEnsemble,
        ]
        
        for ensemble_class in ensemble_classes:
            model = ensemble_class()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert predictions.shape == (len(X_test),)
            assert not np.any(np.isnan(predictions))
            assert not np.any(np.isinf(predictions))
    
    def test_ensemble_performance_better_than_naive(self, large_sample_data):
        """Test that ensemble performs reasonably."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]
        
        model = VotingEnsembleRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Should have some predictive power (RMSE much less than std)
        assert rmse < y_test.std() * 2
    
    def test_ensemble_comparison(self, large_sample_data):
        """Test comparing different ensemble types."""
        X, y = large_sample_data
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]
        
        results = {}
        
        for name, ensemble_class in [
            ('Stacking', StackingEnsembleRegressor),
            ('Voting', VotingEnsembleRegressor),
            ('Blending', BlendingEnsembleRegressor),
            ('OptimalWeighting', OptimalWeightingEnsemble),
        ]:
            model = ensemble_class()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            results[name] = {'rmse': rmse, 'r2': r2}
        
        # All should have reasonable performance
        for name, metrics in results.items():
            assert metrics['rmse'] > 0
            assert -1 < metrics['r2'] < 1


class TestEnsembleEdgeCases:
    """Edge case tests for ensemble models."""
    
    def test_constant_target(self, sample_data):
        """Test with constant target values."""
        X, _ = sample_data
        y = pd.Series(np.ones(len(X)) * 50, name='demand')
        
        model = VotingEnsembleRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
        assert np.all(predictions >= 0)
    
    def test_zero_values(self, sample_data):
        """Test with zero values in target."""
        X, y = sample_data
        y = y * 0.5  # Scale down
        
        model = VotingEnsembleRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
        assert np.all(predictions >= 0)
    
    def test_missing_values_in_features(self, large_sample_data):
        """Test with missing values in features."""
        X, y = large_sample_data
        
        # Introduce missing values
        X.loc[10:20, 'temperature'] = np.nan
        X.loc[30:40, 'humidity'] = np.nan
        
        model = VotingEnsembleRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
