# examples/example_random_forest_models.py
"""
Random Forest Regression Baseline - Complete Examples

This module demonstrates the Random Forest baseline model with 5 comprehensive examples:
1. Basic Random Forest Model - Simple initialization, fitting, and prediction
2. Feature Importance Analysis - Understanding which features drive predictions
3. Hyperparameter Tuning - Comparing different Random Forest configurations
4. Random Forest vs Linear Models - Benchmarking against linear baseline
5. Model Comparison - Testing multiple ensemble approaches

Each example includes:
- Data preparation
- Model training
- Performance metrics (RMSE, MAE, R²)
- Detailed logging output
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

from src.models.baseline.random_forest_models import RandomForestBaseline, RandomForestAdvanced
from src.models.baseline.linear_models import LinearRegressionBaseline
from src.models.baseline.naive_models import SeasonalNaiveModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging to see model behavior
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 336):
    """
    Create realistic retail demand data for examples.
    
    Args:
        n_samples: Number of time steps to generate
        
    Returns:
        X: Feature DataFrame
        y: Target Series
    """
    np.random.seed(42)
    
    # Generate time index (hourly data, 2 weeks)
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Create realistic retail data
    # Note: Including both raw and aggregated features to support feature engineering
    temp_base = 15 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 2, n_samples)
    
    data = {
        'dt': dates,
        'store_id': np.tile(np.arange(1, 6), n_samples // 5 + 1)[:n_samples],
        'product_id': np.tile(np.arange(1, 4), n_samples // 3 + 1)[:n_samples],
        'city_id': np.tile([1, 2], n_samples // 2 + 1)[:n_samples],
        # Raw features
        'temperature': temp_base,
        'humidity': 50 + 20 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 5, n_samples),
        'rainfall': np.random.exponential(1, n_samples),
        'discount': np.random.uniform(0, 0.3, n_samples),
        'stock': 100 + 50 * np.sin(np.arange(n_samples) * 2 * np.pi / 168) + np.random.normal(0, 10, n_samples),
        'day_of_week': np.array([d.weekday() for d in dates]),
        'is_weekend': np.array([1 if d.weekday() >= 5 else 0 for d in dates]),
        'is_holiday': np.random.binomial(1, 0.05, n_samples),
        # Lag and aggregated features
        'sales_lag_1': np.random.uniform(10, 100, n_samples),
        'sales_lag_7': np.random.uniform(10, 100, n_samples),
        'avg_monthly_sales': np.random.uniform(50, 150, n_samples),
        'sale_amount': np.random.exponential(30, n_samples),
        # Aggregated features expected by feature engineering
        'avg_temperature': np.mean(temp_base) + np.random.normal(0, 1, n_samples),
        'precpt': np.random.exponential(1, n_samples),
        'is_evening_rush': (np.array([d.hour for d in dates]) >= 18).astype(int),
        'hours_stock_status': np.random.exponential(50, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Create target with some pattern
    y = (20 + 
         10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) +  # Daily pattern
         5 * np.sin(np.arange(n_samples) * 2 * np.pi / 168) +   # Weekly pattern
         0.5 * X['temperature'] +
         0.3 * X['stock'] +
         np.random.normal(0, 5, n_samples))
    
    y = pd.Series(np.maximum(y, 0))  # Ensure non-negative
    
    return X, y


def example_1_basic_random_forest():
    """
    Example 1: Basic Random Forest Model
    
    Demonstrates:
    - Model initialization with default parameters
    - Fitting the model
    - Generating predictions
    - Basic performance metrics
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Random Forest Model")
    print("="*80)
    
    logger.info("Creating sample data...")
    X, y = create_sample_data(n_samples=336)
    
    # Split into train/test
    split_idx = 280
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Initialize and train model
    logger.info("Initializing Random Forest model...")
    model = RandomForestBaseline()
    
    logger.info("Fitting model...")
    model.fit(X_train, y_train)
    
    # Generate predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✓ Random Forest Model Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"\n  Sample predictions (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        print(f"    Actual: {y_test.iloc[i]:7.2f} → Predicted: {y_pred[i]:7.2f}")
    
    # Get model summary
    summary = model.get_model_summary()
    print(f"\n  Model Summary:")
    print(f"    Model Type: {summary.get('model_type', 'N/A')}")
    print(f"    N Estimators: {summary.get('n_estimators', 'N/A')}")
    print(f"    N Features: {summary.get('n_features', 'N/A')}")


def example_2_feature_importance():
    """
    Example 2: Feature Importance Analysis
    
    Demonstrates:
    - Training a Random Forest model
    - Extracting feature importance scores
    - Identifying most influential features
    - Understanding model behavior
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Feature Importance Analysis")
    print("="*80)
    
    logger.info("Creating sample data...")
    X, y = create_sample_data(n_samples=336)
    
    logger.info("Initializing and training model...")
    model = RandomForestBaseline()
    model.fit(X, y)
    
    # Get feature importance
    logger.info("Extracting feature importance...")
    importance_dict = model.get_feature_importance(top_n=15)
    
    print(f"\n✓ Top 15 Important Features:")
    total_importance = 0
    for i, (feature, importance) in enumerate(importance_dict.items(), 1):
        bar_length = int(importance * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"  {i:2d}. {feature:30s} | {bar} {importance:.4f}")
        total_importance += importance
    
    print(f"\n  Cumulative importance (top 15): {total_importance:.4f}")
    
    # Get all features
    all_importance = model.get_feature_importance_all()
    print(f"  Total features used: {len(all_importance)}")
    
    # Missing value indicators
    missing_indicators = [f for f in model.feature_names if f.startswith('is_missing_')]
    print(f"  Missing value indicators: {len(missing_indicators)}")


def example_3_hyperparameter_tuning():
    """
    Example 3: Hyperparameter Tuning Comparison
    
    Demonstrates:
    - Training multiple Random Forest models with different parameters
    - Comparing performance across configurations
    - Understanding parameter impact
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Hyperparameter Tuning Comparison")
    print("="*80)
    
    logger.info("Creating sample data...")
    X, y = create_sample_data(n_samples=336)
    
    # Split into train/test
    split_idx = 280
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Test different configurations
    configs = [
        {
            'name': 'Shallow (depth=5)',
            'n_estimators': 50,
            'max_depth': 5,
        },
        {
            'name': 'Default (depth=20)',
            'n_estimators': 100,
            'max_depth': 20,
        },
        {
            'name': 'Deep (depth=50)',
            'n_estimators': 100,
            'max_depth': 50,
        },
        {
            'name': 'Large (200 trees, depth=25)',
            'n_estimators': 200,
            'max_depth': 25,
        },
    ]
    
    print(f"\n✓ Training {len(configs)} Random Forest configurations:")
    
    results = []
    for config_spec in configs:
        name = config_spec.pop('name')
        config = {
            'n_estimators': config_spec.get('n_estimators', 100),
            'max_depth': config_spec.get('max_depth', 20),
        }
        
        logger.info(f"Training model: {name}...")
        model = RandomForestBaseline(config=config)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'name': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        })
    
    # Display results
    print(f"\n  Configuration                    │   RMSE  │   MAE   │   R²")
    print(f"  {'-'*70}")
    
    for result in results:
        print(f"  {result['name']:32s} │ {result['rmse']:7.4f} │ {result['mae']:7.4f} │ {result['r2']:7.4f}")
    
    # Find best
    best = min(results, key=lambda x: x['rmse'])
    print(f"\n  ✓ Best RMSE: {best['name']} ({best['rmse']:.4f})")


def example_4_random_forest_vs_linear():
    """
    Example 4: Random Forest vs Linear Models
    
    Demonstrates:
    - Comparing Random Forest with Linear Regression
    - Understanding when each model excels
    - Performance on same dataset
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Random Forest vs Linear Models")
    print("="*80)
    
    logger.info("Creating sample data...")
    X, y = create_sample_data(n_samples=336)
    
    # Split into train/test
    split_idx = 280
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        'Random Forest': RandomForestBaseline(config={'n_estimators': 100}),
        'Linear Regression': LinearRegressionBaseline(),
        'Ridge Regression': LinearRegressionBaseline(config={'alpha': 1.0}),
        'Seasonal Naive': SeasonalNaiveModel(),
    }
    
    print(f"\n✓ Training {len(models)} models for comparison:")
    
    results = {}
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    
    # Display results
    print(f"\n  Model                    │   RMSE  │   MAE   │   R²")
    print(f"  {'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"  {model_name:24s} │ {metrics['rmse']:7.4f} │ {metrics['mae']:7.4f} │ {metrics['r2']:7.4f}")
    
    # Calculate improvements
    rf_rmse = results['Random Forest']['rmse']
    linear_rmse = results['Linear Regression']['rmse']
    improvement = ((linear_rmse - rf_rmse) / linear_rmse) * 100
    
    print(f"\n  ✓ Random Forest {improvement:+.1f}% improvement vs Linear Regression on RMSE")


def example_5_ensemble_comparison():
    """
    Example 5: Ensemble Method Comparison
    
    Demonstrates:
    - Using RandomForestAdvanced for ensemble comparison
    - Testing different ensemble methods
    - Model selection strategies
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Ensemble Method Comparison")
    print("="*80)
    
    logger.info("Creating sample data...")
    X, y = create_sample_data(n_samples=336)
    
    # Split into train/test
    split_idx = 280
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    ensemble_methods = ['random_forest', 'gradient_boosting']
    
    print(f"\n✓ Comparing ensemble methods:")
    
    results = {}
    for method in ensemble_methods:
        try:
            logger.info(f"Training {method}...")
            model = RandomForestAdvanced(ensemble_type=method)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[method] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
            }
            
            print(f"\n  ✓ {method.upper()}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")
            print(f"    R²:   {r2:.4f}")
        
        except Exception as e:
            logger.warning(f"Failed to train {method}: {str(e)}")
            print(f"\n  ✗ {method}: {str(e)}")
    
    # Summary
    if results:
        print(f"\n  Summary:")
        for method, metrics in results.items():
            print(f"    {method}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")


def main():
    """Run all examples."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  RANDOM FOREST REGRESSION BASELINE - COMPREHENSIVE EXAMPLES".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        example_1_basic_random_forest()
        example_2_feature_importance()
        example_3_hyperparameter_tuning()
        example_4_random_forest_vs_linear()
        example_5_ensemble_comparison()
        
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  ✓ ALL EXAMPLES COMPLETED SUCCESSFULLY".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
