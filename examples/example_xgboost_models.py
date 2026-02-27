# examples/example_xgboost_models.py
"""
XGBoost Regression Baseline - Usage Examples

This module demonstrates 5 complete usage patterns:
1. Basic XGBoost model training and prediction
2. Feature importance analysis
3. Hyperparameter tuning and comparison
4. Model comparison across baselines
5. Ensemble methods with custom configurations

All examples use synthetic retail demand data with realistic patterns.
"""

import numpy as np
import pandas as pd
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 280, n_stores: int = 3, n_products: int = 5) -> tuple:
    """
    Create synthetic retail demand dataset.
    
    Args:
        n_samples: Number of time samples (hourly)
        n_stores: Number of stores
        n_products: Number of products per store
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    np.random.seed(42)
    
    # Generate dates (hourly data)
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
    
    # Create data with realistic patterns
    store_ids = np.random.choice(range(1, n_stores + 1), n_samples)
    product_ids = np.random.choice(range(1, n_products + 1), n_samples)
    city_ids = np.random.choice([1, 2], n_samples)
    
    # Generate base demand with patterns
    base_demand = np.ones(n_samples) * 50
    
    # Hour-of-day effect (morning peak, evening peak)
    hour_effect = 30 * np.sin((dates.hour - 6) * np.pi / 12)
    
    # Day-of-week effect (weekends higher)
    day_effect = np.where(dates.dayofweek >= 5, 10, 0)
    
    # Temperature effect (non-linear)
    temperature = 15 + 10 * np.sin((dates.month - 1) * np.pi / 6) + np.random.normal(0, 2, n_samples)
    temp_effect = -0.5 * (temperature - 20) ** 2 + 10
    
    # Humidity
    humidity = 50 + 20 * np.sin((dates.month - 1) * np.pi / 6) + np.random.normal(0, 5, n_samples)
    
    # Discount effect (increases demand)
    discount_rate = np.random.uniform(0, 0.3, n_samples)
    discount_effect = discount_rate * 50
    
    # Competitor pricing
    competitor_price = 25 + np.random.normal(0, 5, n_samples)
    price_effect = -0.5 * competitor_price
    
    # Stock level effect (stockouts reduce sales)
    stock_level = np.random.uniform(0, 1000, n_samples)
    stock_effect = np.where(stock_level < 20, -30, 0)
    
    # Promotion effect
    promotion_active = np.random.binomial(1, 0.2, n_samples)
    promo_effect = promotion_active * 20
    
    # Customer count
    customer_count = 30 + 20 * np.random.exponential(1, n_samples)
    customer_effect = customer_count * 0.5
    
    # Holiday effect
    is_holiday = np.random.binomial(1, 0.05, n_samples)
    holiday_effect = is_holiday * 25
    
    # Holiday weekend interaction
    is_evening_rush = ((dates.hour >= 17) & (dates.hour <= 20)).astype(int)
    evening_effect = is_evening_rush * 15
    
    # Aggregate demand
    sale_amount = (
        base_demand +
        hour_effect +
        day_effect +
        temp_effect +
        discount_effect +
        price_effect +
        stock_effect +
        promo_effect +
        customer_effect +
        holiday_effect +
        evening_effect +
        store_ids * 5 +  # Store-specific baseline
        product_ids * 2 +  # Product-specific baseline
        np.random.normal(0, 10, n_samples)  # Noise
    )
    
    # Ensure positive sales
    sale_amount = np.maximum(sale_amount, 0)
    
    # Aggregated features for feature engineering
    temp_series = pd.Series(temperature)
    avg_temperature = temp_series.rolling(window=12, center=True).mean()
    avg_temperature.fillna(np.mean(temperature), inplace=True)
    avg_temperature = avg_temperature.values
    
    precpt = np.random.exponential(5, n_samples)  # Precipitation
    hours_stock_status = np.where(stock_level > 100, np.random.uniform(8, 24, n_samples), 
                                   np.random.uniform(0, 8, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'store_id': store_ids,
        'product_id': product_ids,
        'city_id': city_ids,
        'dt': dates,
        'temperature': temperature,
        'humidity': humidity,
        'day_of_week': dates.dayofweek,
        'hour': dates.hour,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'is_holiday': is_holiday,
        'discount': discount_rate,  # Note: named 'discount' for feature engineering compatibility
        'competitor_price': competitor_price,
        'stock_level': stock_level,
        'promotion_active': promotion_active,
        'customer_count': customer_count,
        'avg_temperature': avg_temperature,
        'precpt': precpt,
        'is_evening_rush': is_evening_rush,
        'hours_stock_status': hours_stock_status,
    })
    
    X = df
    y = pd.Series(sale_amount, name='sale_amount')
    
    return X, y


def example_1_basic_model():
    """Example 1: Basic XGBoost Model Training and Prediction."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic XGBoost Model")
    print("="*70)
    
    # Import model
    try:
        from src.models.baseline.xgboost_models import XGBoostBaseline
    except ImportError as e:
        print(f"Error importing XGBoostBaseline: {e}")
        print("Make sure XGBoost is installed: pip install xgboost")
        return
    
    # Create data
    X, y = create_sample_data(n_samples=280)
    X_train = X.iloc[:240]
    y_train = y.iloc[:240]
    X_test = X.iloc[240:]
    y_test = y.iloc[240:]
    
    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Create and train model
    print("\nTraining XGBoost model...")
    model = XGBoostBaseline()
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Train MAE:  {train_mae:.4f}")
    print(f"  Test MAE:   {test_mae:.4f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    
    # Sample predictions
    print(f"\nSample predictions (first 5 test samples):")
    print(f"  Actual:     {y_test.iloc[:5].values}")
    print(f"  Predicted:  {y_pred_test[:5]}")
    print(f"  Absolute Error: {np.abs(y_test.iloc[:5].values - y_pred_test[:5])}")
    
    print(f"\n✓ Example 1 completed successfully")


def example_2_feature_importance():
    """Example 2: Feature Importance Analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Feature Importance Analysis")
    print("="*70)
    
    try:
        from src.models.baseline.xgboost_models import XGBoostBaseline
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    # Create and train model
    X, y = create_sample_data(n_samples=280)
    X_train = X.iloc[:240]
    y_train = y.iloc[:240]
    
    print(f"\nTraining model with {X_train.shape[0]} samples...")
    model = XGBoostBaseline()
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = model.get_feature_importance(top_n=15)
    
    print(f"\nTop 15 Important Features:")
    print(f"{'Rank':<6}{'Feature':<40}{'Importance':<12}{'Cumulative':<12}")
    print("-" * 70)
    
    for idx, row in importance_df.iterrows():
        print(f"{idx+1:<6}{row['feature']:<40}{row['importance']:<12.2f}{row['cumulative_importance']:<12.4f}")
    
    # Total features
    all_importance = model.get_feature_importance_all()
    print(f"\nTotal features used: {len(all_importance)}")
    print(f"Missing value indicators: {len([f for f in all_importance['feature'] if 'is_missing' in f])}")
    
    print(f"\n✓ Example 2 completed successfully")


def example_3_hyperparameter_tuning():
    """Example 3: Hyperparameter Tuning and Comparison."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Hyperparameter Tuning")
    print("="*70)
    
    try:
        from src.models.baseline.xgboost_models import XGBoostBaseline
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    from sklearn.metrics import mean_squared_error
    
    # Create data
    X, y = create_sample_data(n_samples=280)
    X_train = X.iloc[:240]
    y_train = y.iloc[:240]
    X_test = X.iloc[240:]
    y_test = y.iloc[240:]
    
    # Define hyperparameter configurations
    configs = [
        {
            'name': 'Shallow (depth=3)',
            'params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
        },
        {
            'name': 'Medium (depth=6)',
            'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        },
        {
            'name': 'Deep (depth=10)',
            'params': {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1}
        },
        {
            'name': 'Boosted (200 trees, depth=8, lr=0.05)',
            'params': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05}
        },
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    print(f"{'Configuration':<40}{'Test RMSE':<12}{'Status'}")
    print("-" * 62)
    
    results = []
    best_rmse = float('inf')
    best_config = None
    
    for config_dict in configs:
        name = config_dict['name']
        params = config_dict['params']
        
        try:
            model = XGBoostBaseline(config=params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results.append({'name': name, 'rmse': rmse, 'params': params})
            
            status = "✓ BEST" if rmse < best_rmse else ""
            print(f"{name:<40}{rmse:<12.4f}{status}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = name
                
        except Exception as e:
            print(f"{name:<40}{'ERROR':<12}{str(e)[:20]}")
    
    print(f"\nBest Configuration: {best_config} (RMSE: {best_rmse:.4f})")
    print(f"\n✓ Example 3 completed successfully")


def example_4_model_comparison():
    """Example 4: XGBoost vs Other Baseline Models."""
    print("\n" + "="*70)
    print("EXAMPLE 4: XGBoost vs Linear/Random Forest Models")
    print("="*70)
    
    try:
        from src.models.baseline.xgboost_models import XGBoostBaseline
        from src.models.baseline.linear_models import LinearRegressionBaseline
        from src.models.baseline.random_forest_models import RandomForestBaseline
        from src.models.baseline.naive_models import SeasonalNaiveModel
    except ImportError as e:
        print(f"Warning: Could not import all models: {e}")
        return
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Create data
    X, y = create_sample_data(n_samples=280)
    X_train = X.iloc[:240]
    y_train = y.iloc[:240]
    X_test = X.iloc[240:]
    y_test = y.iloc[240:]
    
    print(f"\nTraining multiple models...")
    
    models = {
        'XGBoost': XGBoostBaseline(config={'n_estimators': 100}),
        'Random Forest': RandomForestBaseline(config={'n_estimators': 100}),
        'Linear Regression': LinearRegressionBaseline(),
        'Seasonal Naive': SeasonalNaiveModel(),
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"  Training {name}...", end='', flush=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            print(f" ✓")
        except Exception as e:
            print(f" ✗ ({str(e)[:30]})")
    
    # Display results
    print(f"\nModel Comparison Results:")
    print(f"{'Model':<25}{'RMSE':<12}{'MAE':<12}{'R²':<12}{'Status'}")
    print("-" * 65)
    
    best_rmse = min([v['rmse'] for v in results.values()])
    
    for name, metrics in results.items():
        status = "✓ BEST" if metrics['rmse'] == best_rmse else ""
        print(f"{name:<25}{metrics['rmse']:<12.4f}{metrics['mae']:<12.4f}{metrics['r2']:<12.4f}{status}")
    
    # Calculate improvements
    if 'XGBoost' in results and 'Linear Regression' in results:
        xgb_rmse = results['XGBoost']['rmse']
        linear_rmse = results['Linear Regression']['rmse']
        improvement = ((linear_rmse - xgb_rmse) / linear_rmse) * 100
        print(f"\nXGBoost improvement vs Linear: {improvement:+.1f}%")
    
    print(f"\n✓ Example 4 completed successfully")


def example_5_ensemble_methods():
    """Example 5: Ensemble Methods and Custom Configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Ensemble Methods with Custom Configurations")
    print("="*70)
    
    try:
        from src.models.baseline.xgboost_models import XGBoostBaseline, XGBoostAdvanced
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    from sklearn.metrics import mean_squared_error
    
    # Create data
    X, y = create_sample_data(n_samples=280)
    X_train = X.iloc[:240]
    y_train = y.iloc[:240]
    X_test = X.iloc[240:]
    y_test = y.iloc[240:]
    
    print(f"\nTesting ensemble configurations...")
    
    # Configuration 1: Standard XGBoost
    print(f"\n1. Standard XGBoost (100 trees, depth=6)")
    try:
        model1 = XGBoostBaseline(config={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        })
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_test)
        rmse1 = np.sqrt(mean_squared_error(y_test, y_pred1))
        print(f"   Test RMSE: {rmse1:.4f} ✓")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")
    
    # Configuration 2: Regularized XGBoost
    print(f"\n2. Regularized XGBoost (high L2, low learning rate)")
    try:
        model2 = XGBoostBaseline(config={
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        })
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)
        rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
        print(f"   Test RMSE: {rmse2:.4f} ✓")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")
    
    # Configuration 3: Deep XGBoost
    print(f"\n3. Deep XGBoost (deep trees, high subsample)")
    try:
        model3 = XGBoostBaseline(config={
            'n_estimators': 100,
            'max_depth': 12,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        })
        model3.fit(X_train, y_train)
        y_pred3 = model3.predict(X_test)
        rmse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
        print(f"   Test RMSE: {rmse3:.4f} ✓")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")
    
    # XGBoostAdvanced wrapper
    print(f"\n4. Advanced Wrapper (xgboost algorithm)")
    try:
        adv_model = XGBoostAdvanced(algorithm='xgboost')
        adv_model.fit(X_train, y_train)
        y_pred_adv = adv_model.predict(X_test)
        rmse_adv = np.sqrt(mean_squared_error(y_test, y_pred_adv))
        print(f"   Test RMSE: {rmse_adv:.4f} ✓")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")
    
    print(f"\n✓ Example 5 completed successfully")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("XGBoost REGRESSION BASELINE - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    examples = [
        example_1_basic_model,
        example_2_feature_importance,
        example_3_hyperparameter_tuning,
        example_4_model_comparison,
        example_5_ensemble_methods,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("█" * 70)
    print("█" + " " * 16 + "✓ ALL EXAMPLES COMPLETED" + " " * 27 + "█")
    print("█" * 70)
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
