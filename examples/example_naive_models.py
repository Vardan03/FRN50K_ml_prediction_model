#!/usr/bin/env python3
# examples/example_naive_models.py
"""
Example usage of naive forecasting models.

This script demonstrates:
1. Creating and configuring naive models
2. Fitting models on training data
3. Generating predictions
4. Evaluating performance
5. Saving and loading mode
6. Model comparison and analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.naive_models import (
    SimpleNaiveModel,
    SeasonalNaiveModel,
    WeightedMovingAverageModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add file handler for logging to app.log
file_handler = logging.FileHandler('app.log', mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)


# ============================================================================
# Example 1: Simple Naive Model - Basic Usage
# ============================================================================

def example_1_simple_naive_basic():
    """Example 1: Basic Simple Naive Model usage."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Naive Model - Basic Usage")
    print("=" * 70)
    
    # Create synthetic training data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    
    X_train = pd.DataFrame({
        'store_id': [1, 2, 3] * 33 + [1],
        'product_id': [10, 11, 12] * 33 + [10],
        'dt': dates
    })
    y_train = pd.Series(np.random.uniform(0, 10, 100))
    
    # Initialize and fit model
    model = SimpleNaiveModel(config={
        'fill_method': 'forward_fill',
        'default_value': 0.0
    })
    
    print(f"\n1. Fitting model on {len(X_train)} training samples...")
    model.fit(X_train, y_train)
    
    # Display model info
    summary = model.get_model_summary()
    print(f"\n2. Model Summary:")
    print(f"   - Unique items: {summary['unique_items']}")
    print(f"   - Fill method: {summary['fill_method']}")
    print(f"   - Default value: {summary['default_value']}")
    
    # Generate predictions
    print(f"\n3. Making predictions on first 10 samples...")
    X_test = X_train.head(10)
    predictions = model.predict(X_test)
    
    print(f"   Predictions: {predictions[:5]} ... (showing 5 of {len(predictions)})")
    
    # Handle new items
    print(f"\n4. Handling new store-product combinations...")
    X_new = pd.DataFrame({
        'store_id': [999, 999],
        'product_id': [999, 999]
    })
    predictions_new = model.predict(X_new)
    print(f"   Predictions for new items: {predictions_new}")
    print(f"   All using default value: {all(predictions_new == model.default_value)}")


# ============================================================================
# Example 2: Seasonal Naive Model - Weekly Patterns
# ============================================================================

def example_2_seasonal_naive():
    """Example 2: Seasonal Naive Model with weekly patterns."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Seasonal Naive Model - Weekly Patterns")
    print("=" * 70)
    
    # Create training data with clear weekly pattern
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=336, freq='h')  # 2 weeks
    
    X_train = pd.DataFrame({
        'store_id': [1] * 336,
        'product_id': [10] * 336,
        'dt': dates
    })
    
    # Create data with weekly seasonality
    hours = dates.hour
    days = dates.dayofweek
    
    # Sales pattern: higher on weekends, peak hours in afternoon
    daily_pattern = 3 * np.sin((hours - 12) * np.pi / 12)
    weekend_boost = np.where(days >= 5, 5, 0)  # +5 on weekends
    y_train_array = 5 + daily_pattern + weekend_boost + np.random.normal(0, 0.5, 336)
    y_train = pd.Series(np.maximum(y_train_array, 0))
    
    # Fit model
    model = SeasonalNaiveModel(config={
        'seasonal_period': 168,
        'fallback_method': 'simple_naive',
        'default_value': 0.0
    })
    
    print(f"\n1. Fitting model on 2 weeks of hourly data (336 samples)...")
    model.fit(X_train, y_train)
    
    # Display model info
    summary = model.get_model_summary()
    print(f"\n2. Model Summary:")
    print(f"   - Items with seasonal pattern: {summary['seasonal_items']}")
    print(f"   - Items with fallback pattern: {summary['fallback_items']}")
    print(f"   - Seasonal period: {summary['seasonal_period']} hours")
    
    # Make predictions for future week (same pattern as previous week)
    print(f"\n3. Making predictions for future week (same hour-of-week pattern)...")
    future_dates = pd.date_range('2025-01-15', periods=24, freq='h')
    X_test = pd.DataFrame({
        'store_id': [1] * 24,
        'product_id': [10] * 24,
        'dt': future_dates
    })
    predictions = model.predict(X_test)
    
    print(f"   Sample predictions for hours 0-5:")
    for i, pred in enumerate(predictions[:6]):
        print(f"   Hour {i}: {pred:.2f}")
    
    # Compare predictions for same hour in future weeks
    print(f"\n4. Verifying seasonal pattern consistency...")
    monday_0am_pred = predictions[0]
    print(f"   Prediction for Monday 00:00: {monday_0am_pred:.2f}")
    print(f"   (Same hour should have similar pattern in future weeks)")


# ============================================================================
# Example 3: Weighted Moving Average - Trend and Seasonality
# ============================================================================

def example_3_weighted_moving_average():
    """Example 3: Weighted Moving Average with multiple windows."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Weighted Moving Average - Trend & Seasonality")
    print("=" * 70)
    
    # Create training data with trend and seasonality
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='h')
    
    X_train = pd.DataFrame({
        'store_id': [1] * 200,
        'product_id': [10] * 200,
        'dt': dates
    })
    
    # Create data with upward trend + seasonality
    hours = dates.hour
    trend = np.linspace(0, 5, 200)
    seasonality = 2 * np.sin(hours * 2 * np.pi / 24)
    y_train_array = trend + seasonality + np.random.normal(0, 0.3, 200)
    y_train = pd.Series(np.maximum(y_train_array, 0))
    
    # Fit model with exponential weighting
    model = WeightedMovingAverageModel(config={
        'windows': [6, 24, 168],
        'weights': 'exponential',
        'alpha': 0.7,
        'default_value': 0.0
    })
    
    print(f"\n1. Fitting model on 200 hours of data...")
    model.fit(X_train, y_train)
    
    # Display model info
    summary = model.get_model_summary()
    print(f"\n2. Model Summary:")
    print(f"   - Windows: {summary['windows']} hours")
    print(f"   - Weighting method: {summary['weighting_method']}")
    print(f"   - Alpha (decay factor): {summary['alpha']}")
    print(f"   - Unique items: {summary['unique_items']}")
    
    # Show how windows are weighted
    print(f"\n3. Window Weighting (alpha={model.alpha}):")
    alpha = model.alpha
    windows = sorted(model.windows)
    weights_raw = [alpha ** (len(windows) - 1 - i) for i in range(len(windows))]
    total_weight = sum(weights_raw)
    weights_norm = [w / total_weight for w in weights_raw]
    
    for window, raw, norm in zip(windows, weights_raw, weights_norm):
        print(f"   - {window}h window: raw={raw:.3f}, normalized={norm:.1%}")
    
    # Make predictions
    print(f"\n4. Making predictions...")
    X_test = X_train.tail(10)
    predictions = model.predict(X_test)
    
    print(f"   Last 5 predictions: {predictions[-5:]}")
    print(f"   Mean prediction: {np.mean(predictions):.2f}")
    print(f"   Std prediction: {np.std(predictions):.2f}")


# ============================================================================
# Example 4: Model Comparison
# ============================================================================

def example_4_model_comparison():
    """Example 4: Compare all three naive models."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 70)
    
    # Create training data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='h')
    
    # Multiple store-product combinations (500 samples)
    store_ids = np.tile([1, 2, 3], 167)[:500]
    product_ids = np.tile([10, 11, 12], 167)[:500]
    
    X_train = pd.DataFrame({
        'store_id': store_ids,
        'product_id': product_ids,
        'dt': dates
    })
    
    # Create sales data with seasonality and trend
    hours = X_train['dt'].dt.hour.values
    days = X_train['dt'].dt.dayofweek.values
    daily_pattern = 3 * np.sin((hours - 12) * np.pi / 12)
    weekly_pattern = 1 * np.cos(days * 2 * np.pi / 7)
    y_train_array = 5 + daily_pattern + weekly_pattern + np.random.normal(0, 0.5, 500)
    y_train = pd.Series(np.maximum(y_train_array, 0))
    
    # Initialize all models
    models = {
        'Simple Naive': SimpleNaiveModel(),
        'Seasonal Naive': SeasonalNaiveModel(),
        'Weighted MA': WeightedMovingAverageModel()
    }
    
    # Train all models
    print(f"\n1. Training all models on {len(X_train)} samples...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"   ✓ {name} trained")
    
    # Generate predictions
    print(f"\n2. Generating predictions...")
    X_test = X_train.iloc[-100:].reset_index(drop=True)
    y_actual = y_train.iloc[-100:].reset_index(drop=True).values
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    # Compare predictions
    print(f"\n3. Prediction Statistics:")
    print(f"   Actual values - Mean: {np.mean(y_actual):.2f}, Std: {np.std(y_actual):.2f}")
    
    for name, preds in predictions.items():
        mae = np.mean(np.abs(preds - y_actual))
        rmse = np.sqrt(np.mean((preds - y_actual) ** 2))
        print(f"   {name:15} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Visualize predictions vs actual (text-based)
    print(f"\n4. Sample Predictions vs Actual (first 10 samples):")
    print(f"   Index | Actual | Simple | Seasonal | Weighted")
    print(f"   " + "-" * 50)
    for i in range(min(10, len(y_actual))):
        print(f"   {i:5d} | {y_actual[i]:6.2f} | {predictions['Simple Naive'][i]:6.2f} | "
              f"{predictions['Seasonal Naive'][i]:8.2f} | {predictions['Weighted MA'][i]:8.2f}")


# ============================================================================
# Example 5: Model Persistence
# ============================================================================

def example_5_model_persistence():
    """Example 5: Save and load models."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Model Persistence")
    print("=" * 70)
    
    # Create and train a model
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    
    X_train = pd.DataFrame({
        'store_id': [1, 2] * 50,
        'product_id': [10, 11] * 50,
        'dt': dates
    })
    y_train = pd.Series(np.random.uniform(0, 10, 100))
    
    model = SeasonalNaiveModel()
    model.fit(X_train, y_train)
    
    # Save model
    model_path = '/tmp/seasonal_naive_example.joblib'
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
        from src.models.base_model import BaseForecastingModel
        loaded_model = BaseForecastingModel.load_model(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   - Model name: {loaded_model.model_name}")
        print(f"   - Is fitted: {loaded_model.is_fitted}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return
    
    # Use loaded model
    print(f"\n3. Using loaded model for predictions...")
    X_test = X_train.head(5)
    predictions = loaded_model.predict(X_test)
    print(f"   Predictions: {predictions}")
    print(f"   ✓ Loaded model works correctly")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NAIVE FORECASTING MODELS - USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        example_1_simple_naive_basic()
        example_2_seasonal_naive()
        example_3_weighted_moving_average()
        example_4_model_comparison()
        example_5_model_persistence()
        
        print("\n" + "=" * 70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
