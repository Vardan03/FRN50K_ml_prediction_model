# examples/example_ensemble_models.py
"""
Complete examples demonstrating ensemble regression models.

Examples:
1. Basic Stacking Ensemble - Combining multiple base learners
2. Voting Ensemble Aggregation - Weighted averaging
3. Blending Ensemble - Holdout validation approach
4. Optimal Weighting Ensemble - Constrained weight optimization
5. Ensemble Comparison - Performance benchmarking
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.baseline.ensemble_models import (
    StackingEnsembleRegressor,
    VotingEnsembleRegressor,
    BlendingEnsembleRegressor,
    OptimalWeightingEnsemble,
)


def create_sample_data(n_samples: int = 300):
    """
    Create realistic synthetic retail demand data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Feature dataframe
        y: Target series
    """
    np.random.seed(42)
    
    # Date range
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    
    # Base features
    store_id = np.random.randint(1, 5, n_samples)
    product_id = np.random.randint(1, 10, n_samples)
    city_id = np.random.randint(1, 3, n_samples)
    
    # Environmental features
    temperature = 20 + 10 * np.sin(np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 20 * np.sin(np.arange(n_samples) / 12) + np.random.normal(0, 5, n_samples)
    precpt = np.random.exponential(5, n_samples)  # Precipitation
    
    # Time features
    day_of_week = np.array([d.dayofweek for d in dates])
    hour = np.array([d.hour for d in dates])
    is_holiday = (day_of_week >= 5).astype(int)
    
    # Marketing and business features
    discount = np.random.uniform(0, 0.5, n_samples)
    marketing_spend = np.random.exponential(100, n_samples)
    competitor_price = 100 + np.random.normal(0, 10, n_samples)
    stock_level = np.random.uniform(100, 1000, n_samples)
    traffic = 500 + 200 * np.sin(np.arange(n_samples) / 20) + np.random.normal(0, 50, n_samples)
    conversion_rate = 0.1 + 0.05 * np.sin(np.arange(n_samples) / 30) + np.random.normal(0, 0.02, n_samples)
    
    # Customer features
    avg_order_value = 50 + np.random.normal(0, 10, n_samples)
    customer_satisfaction = 3.5 + np.random.normal(0, 0.5, n_samples)
    repeat_customers = np.random.uniform(0, 1, n_samples)
    avg_temperature = 20 + 10 * np.sin(np.arange(n_samples) / 24)  # Added for feature engineering
    
    # Seasonal
    seasonal_index = 0.8 + 0.4 * np.sin(np.arange(n_samples) * 2 * np.pi / 168)
    
    # Sales (proxy for initial demand pattern)
    sales_amount = 100 + 50 * np.sin(np.arange(n_samples) / 25) + np.random.normal(0, 10, n_samples)
    hours_stock_status = np.random.uniform(0, 24, n_samples)
    is_weekend = is_holiday.copy()
    
    # Create dataframe with all required columns
    X = pd.DataFrame({
        'store_id': store_id,
        'product_id': product_id,
        'city_id': city_id,
        'dt': dates,
        'discount': discount,
        'temperature': temperature,
        'humidity': humidity,
        'precpt': precpt,
        'day_of_week': day_of_week,
        'hour': hour,
        'is_holiday': is_holiday,
        'is_weekend': is_weekend,
        'marketing_spend': marketing_spend,
        'competitor_price': competitor_price,
        'stock_level': stock_level,
        'hours_stock_status': hours_stock_status,
        'traffic': traffic,
        'conversion_rate': conversion_rate,
        'avg_order_value': avg_order_value,
        'customer_satisfaction': customer_satisfaction,
        'repeat_customers': repeat_customers,
        'avg_temperature': avg_temperature,
        'seasonal_index': seasonal_index,
        'sales_amount': sales_amount,
    })
    
    # Target: demand (non-negative)
    y = (
        30 +
        15 * np.sin(np.arange(n_samples) / 20) +  # Trend
        10 * np.cos(np.arange(n_samples) * 2 * np.pi / 168) +  # Weekly pattern
        0.02 * marketing_spend +  # Marketing impact
        -5 * discount +  # Discount impact
        3 * customer_satisfaction +  # Satisfaction impact
        np.random.normal(0, 5, n_samples)  # Noise
    )
    y = np.maximum(y, 0)  # Ensure non-negative
    
    y = pd.Series(y, name='demand')
    
    return X, y


def example_1_stacking_ensemble():
    """Example 1: Basic Stacking Ensemble with cross-validation."""
    print("\n" + "="*80)
    print("Example 1: Stacking Ensemble with Cross-Validation")
    print("="*80)
    
    # Generate data
    X, y = create_sample_data(n_samples=300)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    # Initialize and train model
    print("\nInitializing StackingEnsembleRegressor...")
    model = StackingEnsembleRegressor(config={
        'cv_splits': 5,
        'meta_learner_type': 'ridge',
        'enforce_non_negative': True,
    })
    
    print("Training stacking ensemble...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n✓ Training Performance:")
    print(f"  RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
    print(f"\n✓ Test Performance:")
    print(f"  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\n✓ Model Summary:")
    print(f"  Base Learners: {', '.join(summary['base_learners'])}")
    print(f"  Meta-Learner: {summary['meta_learner']}")
    print(f"  CV Splits: {summary['cv_splits']}")


def example_2_voting_ensemble():
    """Example 2: Voting Ensemble with different aggregation methods."""
    print("\n" + "="*80)
    print("Example 2: Voting Ensemble with Aggregation Methods")
    print("="*80)
    
    # Generate data
    X, y = create_sample_data(n_samples=300)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    aggregation_methods = ['mean', 'median']
    results = {}
    
    for agg_method in aggregation_methods:
        print(f"\nTesting aggregation method: {agg_method}")
        
        config = {
            'aggregation': agg_method,
            'enforce_non_negative': True,
        }
        
        if agg_method == 'weighted_mean':
            config['weights'] = [0.4, 0.35, 0.15]  # Prioritize some models
        
        model = VotingEnsembleRegressor(config=config)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[agg_method] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    print(f"\n✓ Best Performance: {min(results, key=lambda k: results[k]['rmse'])}")


def example_3_blending_ensemble():
    """Example 3: Blending Ensemble with holdout validation."""
    print("\n" + "="*80)
    print("Example 3: Blending Ensemble with Holdout Validation")
    print("="*80)
    
    # Generate data
    X, y = create_sample_data(n_samples=300)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    holdout_ratios = [0.1, 0.2, 0.3]
    results = {}
    
    for ratio in holdout_ratios:
        print(f"\nTesting holdout ratio: {ratio}")
        
        config = {
            'holdout_ratio': ratio,
            'meta_learner_type': 'ridge',
            'enforce_non_negative': True,
        }
        
        model = BlendingEnsembleRegressor(config=config)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[ratio] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")


def example_4_optimal_weighting():
    """Example 4: Optimal Weighting Ensemble with constrained optimization."""
    print("\n" + "="*80)
    print("Example 4: Optimal Weighting Ensemble")
    print("="*80)
    
    # Generate data
    X, y = create_sample_data(n_samples=300)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    # Initialize and train
    print("\nInitializing OptimalWeightingEnsemble...")
    model = OptimalWeightingEnsemble(config={
        'enforce_non_negative': True,
    })
    
    print("Fitting with constrained optimization...")
    model.fit(X_train, y_train)
    
    # Extract weights
    print("\n✓ Optimal Weights:")
    for i, weight in enumerate(model.optimal_weights):
        print(f"  Model {i}: {weight:.4f}")
    
    print(f"\n✓ Weight sum: {np.sum(model.optimal_weights):.4f} (should be 1.0)")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n✓ Performance:")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")


def example_5_ensemble_comparison():
    """Example 5: Compare all ensemble methods."""
    print("\n" + "="*80)
    print("Example 5: Comprehensive Ensemble Comparison")
    print("="*80)
    
    # Generate data
    X, y = create_sample_data(n_samples=300)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    ensemble_configs = {
        'Stacking (Ridge)': (StackingEnsembleRegressor, {'meta_learner_type': 'ridge'}),
        'Stacking (RF)': (StackingEnsembleRegressor, {'meta_learner_type': 'rf'}),
        'Voting (Mean)': (VotingEnsembleRegressor, {'aggregation': 'mean'}),
        'Voting (Median)': (VotingEnsembleRegressor, {'aggregation': 'median'}),
        'Blending': (BlendingEnsembleRegressor, {'holdout_ratio': 0.2}),
        'Optimal Weighting': (OptimalWeightingEnsemble, {}),
    }
    
    results = {}
    
    print("\nTraining and evaluating ensemble models...\n")
    
    for name, (model_class, config) in ensemble_configs.items():
        try:
            print(f"Training {name}...")
            
            model = model_class(config=config)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
            }
            
            print(f"  ✓ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:50]}")
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('RMSE')
    
    print(df_results.to_string())
    
    print("\n✓ Best Performing: " + df_results.index[0])
    print(f"  RMSE: {df_results['RMSE'].iloc[0]:.2f}")


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# ENSEMBLE REGRESSION MODELS - COMPLETE EXAMPLES")
    print("#"*80)
    
    try:
        example_1_stacking_ensemble()
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
    
    try:
        example_2_voting_ensemble()
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
    
    try:
        example_3_blending_ensemble()
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
    
    try:
        example_4_optimal_weighting()
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
    
    try:
        example_5_ensemble_comparison()
    except Exception as e:
        print(f"✗ Example 5 failed: {e}")
    
    print("\n" + "#"*80)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
