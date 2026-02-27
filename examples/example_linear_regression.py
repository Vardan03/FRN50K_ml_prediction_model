#!/usr/bin/env python3
# examples/example_linear_regression.py
"""
Example usage of Linear Regression Baseline for retail demand forecasting.

This script demonstrates:
1. Loading and preparing data
2. Feature engineering for linear models
3. Training linear regression models
4. Making predictions
5. Model evaluation and comparison with naive models
6. Saving and loading models
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.linear_models import LinearRegressionBaseline, LinearForecastingModel
from src.models.baseline.naive_models import SeasonalNaiveModel
from src.data.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Linear Regression Model
# ============================================================================

def example_1_basic_linear_model():
    """Example 1: Basic Linear Regression Model usage."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Linear Regression Model")
    print("=" * 80)
    
    # Create synthetic training data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='h')
    
    # Generate store and product combinations to match 200 samples
    store_ids = np.tile([1, 2, 3], 67)[:200]
    product_ids = np.tile([10, 11, 12], 67)[:200]
    
    X_train = pd.DataFrame({
        'store_id': store_ids,
        'product_id': product_ids,
        'city_id': np.tile([1, 1, 2], 67)[:200],
        'dt': dates,
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
        'sale_amount': np.random.uniform(0.5, 10, 200),  # Placeholder, will be overwritten with y
    })
    
    # Create target with trend and seasonality
    y_train = pd.Series(
        5 + np.linspace(0, 2, 200) + 2*np.sin(np.arange(200)*np.pi/24) + 
        np.random.normal(0, 0.5, 200)
    )
    y_train = np.maximum(y_train, 0)  # Ensure non-negative
    
    # Initialize and fit model
    model = LinearRegressionBaseline(config={
        'alpha': 1.0,
        'use_ridge': True,
        'enforce_non_negative': True
    })
    
    print(f"\n1. Fitting model on {len(X_train)} training samples...")
    model.fit(X_train, y_train)
    
    # Display model info
    summary = model.get_model_summary()
    print(f"\n2. Model Summary:")
    print(f"   - Model type: {summary.get('model_type')}")
    print(f"   - Regularization (alpha): {summary.get('alpha')}")
    print(f"   - Number of features: {summary.get('n_features')}")
    print(f"   - Is fitted: {summary.get('is_fitted')}")
    
    # Generate predictions
    print(f"\n3. Making predictions on test set...")
    X_test = X_train.iloc[-50:]
    y_actual = y_train.iloc[-50:].values
    predictions = model.predict(X_test)
    
    print(f"   Sample predictions (first 10):")
    for i in range(10):
        print(f"   - Actual: {y_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((predictions - y_actual) ** 2))
    mae = np.mean(np.abs(predictions - y_actual))
    print(f"\n4. Model Performance:")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - Mean prediction: {predictions.mean():.4f}")


# ============================================================================
# Example 2: Feature Engineering Impact
# ============================================================================

def example_2_feature_engineering():
    """Example 2: Impact of feature engineering."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Feature Engineering Impact")
    print("=" * 80)
    
    # Create training data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=336, freq='h')  # 2 weeks
    
    X_train = pd.DataFrame({
        'store_id': [1] * 336,
        'product_id': [10] * 336,
        'city_id': [1] * 336,
        'dt': dates,
        'discount': np.random.uniform(0, 0.3, 336),
        'holiday_flag': np.random.randint(0, 2, 336),
        'activity_flag': np.random.randint(0, 2, 336),
        'precpt': np.random.uniform(0, 10, 336),
        'avg_temperature': 20 + 5*np.sin(np.arange(336)*2*np.pi/24),
        'avg_humidity': np.random.uniform(30, 80, 336),
        'avg_wind_level': np.random.uniform(0, 20, 336),
        'hours_sale': dates.hour,
        'hours_stock_status': dates.hour,
        'stock_hour6_22_cnt': np.random.randint(0, 20, 336),
        'sale_amount': np.random.uniform(0.5, 10, 336),
    })
    
    # Create target with clear daily and weekly patterns
    hours = dates.hour
    days = dates.dayofweek
    y_train = pd.Series(
        5 + 2*np.sin((hours-12)*np.pi/12) + np.where(days >= 5, 2, 0) +
        np.random.normal(0, 0.3, 336)
    )
    y_train = np.maximum(y_train, 0)
    
    # Feature engineer
    config = {
        'lag_periods': [1, 7, 168],
        'rolling_windows': [6, 24, 168],
        'seasonal_periods': [24, 168],
    }
    engineer = FeatureEngineer(config=config)
    
    print(f"\n1. Original data shape: {X_train.shape}")
    
    X_engineered = engineer.engineer_all_features(X_train)
    print(f"2. After feature engineering: {X_engineered.shape}")
    print(f"   Features added: {X_engineered.shape[1] - X_train.shape[1]}")
    
    # Show temporal features
    temporal_features = [c for c in X_engineered.columns 
                        if any(x in c for x in ['hour', 'day', 'week', 'sin', 'cos'])]
    print(f"\n3. Temporal features created ({len(temporal_features)}):")
    for feat in temporal_features[:8]:
        print(f"   - {feat}")
    
    # Show lag and rolling features
    lag_features = [c for c in X_engineered.columns if 'lag' in c]
    rolling_features = [c for c in X_engineered.columns if 'rolling' in c]
    
    print(f"\n4. Lag features created ({len(lag_features)}):")
    for feat in lag_features[:5]:
        print(f"   - {feat}")
    
    print(f"\n5. Rolling features created ({len(rolling_features)}):")
    for feat in rolling_features[:5]:
        print(f"   - {feat}")


# ============================================================================
# Example 3: Ridge vs Linear Regression
# ============================================================================

def example_3_ridge_vs_linear():
    """Example 3: Comparing Ridge and Linear Regression."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Ridge vs Linear Regression")
    print("=" * 80)
    
    # Create training data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=300, freq='h')
    
    X_train = pd.DataFrame({
        'store_id': [1, 2] * 150,
        'product_id': [10, 11] * 150,
        'city_id': [1] * 300,
        'dt': dates,
        'discount': np.random.uniform(0, 0.3, 300),
        'holiday_flag': np.random.randint(0, 2, 300),
        'activity_flag': np.random.randint(0, 2, 300),
        'precpt': np.random.uniform(0, 10, 300),
        'avg_temperature': np.random.uniform(15, 30, 300),
        'avg_humidity': np.random.uniform(30, 80, 300),
        'avg_wind_level': np.random.uniform(0, 20, 300),
        'hours_sale': dates.hour,
        'hours_stock_status': dates.hour,
        'stock_hour6_22_cnt': np.random.randint(0, 20, 300),
        'sale_amount': np.random.uniform(0.5, 10, 300),
    })
    
    y_train = pd.Series(np.random.uniform(1, 10, 300))
    
    # Train both models
    models = {
        'Ridge (α=0.1)': LinearRegressionBaseline(config={'alpha': 0.1, 'use_ridge': True}),
        'Ridge (α=1.0)': LinearRegressionBaseline(config={'alpha': 1.0, 'use_ridge': True}),
        'Ridge (α=10.0)': LinearRegressionBaseline(config={'alpha': 10.0, 'use_ridge': True}),
        'Linear (no reg.)': LinearRegressionBaseline(config={'use_ridge': False}),
    }
    
    print(f"\n1. Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"   ✓ {name} trained")
    
    # Compare predictions
    print(f"\n2. Comparing predictions on test set...")
    X_test = X_train.iloc[-50:]
    y_test = y_train.iloc[-50:].values
    
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = np.mean(np.abs(predictions - y_test))
        results[name] = {'rmse': rmse, 'mae': mae}
    
    print(f"\n3. Performance Metrics:")
    print(f"   Model               | RMSE   | MAE")
    print(f"   {'-'*50}")
    for name in results:
        print(f"   {name:20} | {results[name]['rmse']:6.4f} | {results[name]['mae']:6.4f}")


# ============================================================================
# Example 4: Linear vs Naive Models Comparison
# ============================================================================

def example_4_linear_vs_naive():
    """Example 4: Compare Linear Regression with Naive Models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Linear Regression vs Naive Models")
    print("=" * 80)
    
    # Create training data with clear seasonal pattern
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='h')
    
    X_train = pd.DataFrame({
        'store_id': np.tile([1, 2, 3], 167)[:500],
        'product_id': np.tile([10, 11, 12], 167)[:500],
        'city_id': np.tile([1, 1, 2], 167)[:500],
        'dt': dates,
        'discount': np.random.uniform(0, 0.3, 500),
        'holiday_flag': np.random.randint(0, 2, 500),
        'activity_flag': np.random.randint(0, 2, 500),
        'precpt': np.random.uniform(0, 10, 500),
        'avg_temperature': 20 + 5*np.sin(np.arange(500)*2*np.pi/24),
        'avg_humidity': np.random.uniform(30, 80, 500),
        'avg_wind_level': np.random.uniform(0, 20, 500),
        'hours_sale': dates.hour,
        'hours_stock_status': dates.hour,
        'stock_hour6_22_cnt': np.random.randint(0, 20, 500),
        'sale_amount': np.random.uniform(0.5, 10, 500),
    })
    
    # Create target with seasonality
    hours = dates.hour
    days = dates.dayofweek
    y_train = pd.Series(
        5 + 3*np.sin((hours-12)*np.pi/12) + np.where(days >= 5, 1.5, 0) +
        np.random.normal(0, 0.4, 500)
    )
    y_train = np.maximum(y_train, 0)
    
    # Train models
    print(f"\n1. Training Linear and Naive models...")
    
    models = {
        'Linear Regression': LinearRegressionBaseline(),
        'Seasonal Naive': SeasonalNaiveModel(),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"   ✓ {name} trained")
    
    # Test on holdout set
    print(f"\n2. Evaluating on test set...")
    X_test = X_train.iloc[-100:].reset_index(drop=True)
    y_test = y_train.iloc[-100:].reset_index(drop=True).values
    
    print(f"\n3. Performance Comparison:")
    print(f"   Model              | RMSE   | MAE")
    print(f"   {'-'*50}")
    
    for name, model in models.items():
        predictions = model.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = np.mean(np.abs(predictions - y_test))
        
        print(f"   {name:18} | {rmse:6.4f} | {mae:6.4f}")
    
    # Show some predictions
    print(f"\n4. Sample Predictions vs Actual (first 10 samples):")
    print(f"   Index | Actual | Linear | Seasonal")
    print(f"   {'-'*45}")
    
    linear_preds = models['Linear Regression'].predict(X_test)
    naive_preds = models['Seasonal Naive'].predict(X_test)
    
    for i in range(10):
        print(f"   {i:5} | {y_test[i]:6.2f} | {linear_preds[i]:6.2f} | {naive_preds[i]:8.2f}")


# ============================================================================
# Example 5: Model Persistence
# ============================================================================

def example_5_model_persistence():
    """Example 5: Save and load models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Model Persistence (Save/Load)")
    print("=" * 80)
    
    # Create and train a model
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    
    X_train = pd.DataFrame({
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
        'hours_sale': dates.hour,
        'hours_stock_status': dates.hour,
        'stock_hour6_22_cnt': np.random.randint(0, 20, 100),
        'sale_amount': np.random.uniform(0.5, 10, 100),
    })
    
    y_train = pd.Series(np.random.uniform(1, 10, 100))
    
    model = LinearRegressionBaseline()
    model.fit(X_train, y_train)
    
    # Save model
    model_path = '/tmp/linear_regression_example.joblib'
    print(f"\n1. Saving model to {model_path}...")
    try:
        model.save_model(model_path)
        print(f"   ✓ Model saved successfully")
    except Exception as e:
        print(f"   ✗ Failed to save: {e}")
        return
    
    # Load model
    print(f"\n2. Loading model from {model_path}...")
    try:
        loaded_model = LinearRegressionBaseline()
        loaded_model.load_model(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   - Model name: {loaded_model.model_name}")
        print(f"   - Is fitted: {loaded_model.is_fitted}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return
    
    # Use loaded model for predictions
    print(f"\n3. Using loaded model for predictions...")
    X_test = X_train.head(5)
    predictions = loaded_model.predict(X_test)
    print(f"   Predictions: {predictions}")
    print(f"   ✓ Loaded model works correctly")


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION BASELINE - USAGE EXAMPLES")
    print("=" * 80)
    
    try:
        example_1_basic_linear_model()
        example_2_feature_engineering()
        example_3_ridge_vs_linear()
        example_4_linear_vs_naive()
        example_5_model_persistence()
        
        print("\n" + "=" * 80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
