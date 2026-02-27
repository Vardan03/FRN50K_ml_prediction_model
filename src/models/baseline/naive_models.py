# src/models/baseline/naive_models.py
"""
Naive forecasting models for time series prediction.

These simple baseline models provide important benchmarks:
1. Simple Naive: Use last observed value
2. Seasonal Naive: Use value from same hour last week
3. Weighted Moving Average: Use recent historical average with exponential decay

Teaching concepts:
- Baseline models are essential for benchmarking more complex approaches
- These methods are fast and interpretable
- Seasonal patterns are crucial for hourly forecasting
- Proper handling of edge cases is critical in production systems
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Tuple, List
from pathlib import Path

from src.models.base_model import BaseForecastingModel

logger = logging.getLogger(__name__)


class SimpleNaiveModel(BaseForecastingModel):
    """
    Simple Naive forecasting: use the last observed value as prediction.
    
    Key features:
    - Baseline model for comparison
    - Fast inference
    - Handles missing values gracefully
    - Manages new store-product combinations
    
    Assumptions:
    - No significant trend in short-term data
    - Recent observations are representative of immediate future
    - Missing values should be filled forward
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the naive forecaster.
        
        Args:
            config: Dictionary with optional parameters:
                - fill_method: 'forward_fill', 'backward_fill', 'mean' (default: 'forward_fill')
                - default_value: Value to use for new items (default: 0.0)
        """
        super().__init__(model_name="SimpleNaive", config=config)
        self.fill_method = self.config.get('fill_method', 'forward_fill')
        self.default_value = self.config.get('default_value', 0.0)
        
        # Store last values indexed by (store_id, product_id)
        self.last_values = {}
        self.training_data = None
        self.feature_names_list = ['store_id', 'product_id']
        
        logger.info(f"SimpleNaiveModel initialized with fill_method={self.fill_method}, "
                   f"default_value={self.default_value}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SimpleNaiveModel':
        """
        Fit the model by storing last values for each store-product combination.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id', 'dt']
            y: Target values (sale_amount)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting SimpleNaiveModel on {len(X)} samples")
        
        # Validate input
        self.validate_input(X, y)
        
        # Combine X and y for processing
        data = X.copy()
        data['sale_amount'] = y.values
        
        # Store for reference
        self.training_data = data
        
        # Sort by time to ensure we get the last observed value
        if 'dt' in data.columns:
            data = data.sort_values('dt')
        
        # Group by store-product and get the last value
        required_cols = ['store_id', 'product_id']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"X must contain columns: {required_cols}")
        
        try:
            # For each store-product combination, store the last value
            for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
                last_value = group['sale_amount'].iloc[-1]
                
                # Handle missing values in the last value
                if pd.isna(last_value):
                    if self.fill_method == 'forward_fill':
                        last_value = group['sale_amount'].fillna(method='ffill').iloc[-1]
                    elif self.fill_method == 'backward_fill':
                        last_value = group['sale_amount'].fillna(method='bfill').iloc[-1]
                    elif self.fill_method == 'mean':
                        last_value = group['sale_amount'].mean()
                    
                    # If still NaN, use default
                    if pd.isna(last_value):
                        last_value = self.default_value
                
                self.last_values[(int(store_id), int(product_id))] = float(last_value)
            
            logger.info(f"Stored last values for {len(self.last_values)} store-product combinations")
            
        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise
        
        self.is_fitted = True
        self.feature_names = self.feature_names_list
        self.training_history['fit_samples'] = len(X)
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the last observed value for each store-product.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id']
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.debug(f"Making predictions for {len(X)} samples")
        
        predictions = []
        new_items_count = 0
        
        for idx, row in X.iterrows():
            store_id = int(row['store_id'])
            product_id = int(row['product_id'])
            
            key = (store_id, product_id)
            
            if key in self.last_values:
                pred = self.last_values[key]
            else:
                pred = self.default_value
                new_items_count += 1
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if new_items_count > 0:
            logger.warning(f"Found {new_items_count} new store-product combinations, "
                          f"using default value: {self.default_value}")
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of the model."""
        summary = super().get_model_summary()
        summary.update({
            'unique_items': len(self.last_values),
            'fill_method': self.fill_method,
            'default_value': self.default_value
        })
        return summary

class SeasonalNaiveModel(BaseForecastingModel):
    """
    Seasonal Naive forecasting: use value from same hour last week (168-hour cycle).
    
    Key features:
    - Captures weekly seasonality (crucial for retail/commerce data)
    - Uses 168-hour (7-day) seasonal cycle
    - Handles edge cases when historical data insufficient
    - Falls back to simple naive for insufficient data
    
    Assumptions:
    - Weekly patterns are consistent and repeating
    - Same hour in previous week is best predictor
    - At least 168 hours of history available
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Seasonal Naive model.
        
        Args:
            config: Dictionary with optional parameters:
                - seasonal_period: Hours in seasonal cycle (default: 168 = 1 week)
                - fallback_method: 'simple_naive' or 'mean' (default: 'simple_naive')
                - default_value: Value for new items (default: 0.0)
        """
        super().__init__(model_name="SeasonalNaive", config=config)
        self.seasonal_period = self.config.get('seasonal_period', 168)  # 7 days * 24 hours
        self.fallback_method = self.config.get('fallback_method', 'simple_naive')
        self.default_value = self.config.get('default_value', 0.0)
        
        # Store historical hourly values indexed by (store_id, product_id, hour_in_week)
        self.seasonal_values = {}
        self.fallback_values = {}  # For items without enough history
        self.training_data = None
        self.feature_names_list = ['store_id', 'product_id', 'dt']
        
        logger.info(f"SeasonalNaiveModel initialized with seasonal_period={self.seasonal_period}h, "
                   f"fallback_method={self.fallback_method}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SeasonalNaiveModel':
        """
        Fit the model by extracting seasonal patterns.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id', 'dt']
            y: Target values (sale_amount)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting SeasonalNaiveModel on {len(X)} samples with seasonal_period={self.seasonal_period}")
        
        # Validate input
        self.validate_input(X, y)
        
        if 'dt' not in X.columns:
            raise ValueError("X must contain 'dt' column for temporal information")
        
        # Combine X and y
        data = X.copy()
        data['sale_amount'] = y.values
        data['dt'] = pd.to_datetime(data['dt'])
        
        # Store for reference
        self.training_data = data
        
        # Sort by time
        data = data.sort_values('dt')
        
        required_cols = ['store_id', 'product_id']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"X must contain columns: {required_cols}")
        
        try:
            # Extract hour of week (0-167) for seasonal indexing
            data['hour_of_week'] = (data['dt'].dt.dayofweek * 24 + data['dt'].dt.hour)
            
            # For each store-product combination, aggregate values
            # (typically by mean to get representative value for that hour)
            for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
                store_product_key = (int(store_id), int(product_id))
                
                # Create dictionary for this store-product's seasonal values
                seasonal_dict = {}
                
                for hour_of_week, hour_group in group.groupby('hour_of_week'):
                    # Use median to be robust to outliers
                    seasonal_value = hour_group['sale_amount'].median()
                    
                    if pd.isna(seasonal_value):
                        seasonal_value = hour_group['sale_amount'].mean()
                    
                    if pd.isna(seasonal_value):
                        seasonal_value = self.default_value
                    
                    seasonal_dict[int(hour_of_week)] = float(seasonal_value)
                
                # Check if we have enough data (at least one complete week)
                if len(seasonal_dict) >= self.seasonal_period:
                    self.seasonal_values[store_product_key] = seasonal_dict
                else:
                    # Not enough seasonal data, use fallback (last value)
                    last_value = group['sale_amount'].iloc[-1]
                    if pd.isna(last_value):
                        last_value = group['sale_amount'].mean()
                    if pd.isna(last_value):
                        last_value = self.default_value
                    
                    self.fallback_values[store_product_key] = float(last_value)
            
            logger.info(f"Seasonal values extracted for {len(self.seasonal_values)} items, "
                       f"fallback values for {len(self.fallback_values)} items")
            
        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise
        
        self.is_fitted = True
        self.feature_names = self.feature_names_list
        self.training_history['fit_samples'] = len(X)
        self.training_history['seasonal_items'] = len(self.seasonal_values)
        self.training_history['fallback_items'] = len(self.fallback_values)
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using seasonal patterns.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id', 'dt']
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if 'dt' not in X.columns:
            raise ValueError("X must contain 'dt' column for temporal information")
        
        logger.debug(f"Making seasonal predictions for {len(X)} samples")
        
        predictions = []
        seasonal_used = 0
        fallback_used = 0
        default_used = 0
        
        for idx, row in X.iterrows():
            store_id = int(row['store_id'])
            product_id = int(row['product_id'])
            dt = pd.to_datetime(row['dt'])
            
            store_product_key = (store_id, product_id)
            
            # Calculate hour of week for this prediction
            hour_of_week = dt.dayofweek * 24 + dt.hour
            
            if store_product_key in self.seasonal_values:
                seasonal_dict = self.seasonal_values[store_product_key]
                
                # Try to get value for this hour of week
                if hour_of_week in seasonal_dict:
                    pred = seasonal_dict[hour_of_week]
                    seasonal_used += 1
                else:
                    # Hour not in training data, use average
                    pred = np.mean(list(seasonal_dict.values()))
                    seasonal_used += 1
            
            elif store_product_key in self.fallback_values:
                pred = self.fallback_values[store_product_key]
                fallback_used += 1
            
            else:
                # New store-product combination
                pred = self.default_value
                default_used += 1
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if fallback_used > 0 or default_used > 0:
            logger.warning(f"Prediction breakdown - Seasonal: {seasonal_used}, "
                          f"Fallback: {fallback_used}, Default: {default_used}")
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of the model."""
        summary = super().get_model_summary()
        summary.update({
            'seasonal_items': len(self.seasonal_values),
            'fallback_items': len(self.fallback_values),
            'seasonal_period': self.seasonal_period,
            'fallback_method': self.fallback_method
        })
        return summary


class WeightedMovingAverageModel(BaseForecastingModel):
    """
    Weighted Moving Average with exponential decay for recent values.
    
    Key features:
    - Multiple window sizes: 6h, 24h, 168h (1 week)
    - Exponential weighting (recent values weighted more heavily)
    - Smooth predictions without overfitting
    - Handles missing values with interpolation
    
    Assumptions:
    - Recent values are more predictive than older ones
    - Short-term trends matter more than long-term
    - Exponential decay properly captures temporal decay
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Weighted Moving Average model.
        
        Args:
            config: Dictionary with optional parameters:
                - windows: List of window sizes in hours (default: [6, 24, 168])
                - weights: How to combine windows: 'equal', 'exponential' (default: 'exponential')
                - default_value: Value for new items (default: 0.0)
                - alpha: Decay factor for exponential weighting (default: 0.7)
        """
        super().__init__(model_name="WeightedMovingAverage", config=config)
        self.windows = self.config.get('windows', [6, 24, 168])  # Hours
        self.weighting_method = self.config.get('weights', 'exponential')
        self.default_value = self.config.get('default_value', 0.0)
        self.alpha = self.config.get('alpha', 0.7)  # Decay factor
        
        # Store mean values indexed by (store_id, product_id, window_size)
        self.historical_data = {}
        self.training_data = None
        self.feature_names_list = ['store_id', 'product_id', 'dt']
        
        logger.info(f"WeightedMovingAverageModel initialized with windows={self.windows}h, "
                   f"weighting_method={self.weighting_method}, alpha={self.alpha}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'WeightedMovingAverageModel':
        """
        Fit the model by computing statistics for different windows.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id', 'dt']
            y: Target values (sale_amount)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting WeightedMovingAverageModel on {len(X)} samples")
        
        # Validate input
        self.validate_input(X, y)
        
        if 'dt' not in X.columns:
            raise ValueError("X must contain 'dt' column for temporal information")
        
        # Combine X and y
        data = X.copy()
        data['sale_amount'] = y.values
        data['dt'] = pd.to_datetime(data['dt'])
        
        # Store for reference
        self.training_data = data
        
        # Sort by time
        data = data.sort_values('dt')
        
        required_cols = ['store_id', 'product_id']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"X must contain columns: {required_cols}")
        
        try:
            # For each store-product combination, compute statistics
            for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
                store_product_key = (int(store_id), int(product_id))
                
                # Sort by time to ensure chronological order
                group = group.sort_values('dt').reset_index(drop=True)
                
                window_stats = {}
                
                for window in self.windows:
                    # Convert window from hours to number of samples
                    # Assuming hourly data (1 sample per hour)
                    window_samples = window
                    
                    values = []
                    for i in range(max(0, len(group) - window_samples), len(group)):
                        val = group.iloc[i]['sale_amount']
                        if not pd.isna(val):
                            values.append(val)
                    
                    if values:
                        window_stats[window] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'count': len(values)
                        }
                    else:
                        window_stats[window] = {
                            'mean': self.default_value,
                            'std': 0.0,
                            'count': 0
                        }
                
                self.historical_data[store_product_key] = window_stats
            
            logger.info(f"Window statistics computed for {len(self.historical_data)} store-product combinations")
            
        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise
        
        self.is_fitted = True
        self.feature_names = self.feature_names_list
        self.training_history['fit_samples'] = len(X)
        self.training_history['unique_items'] = len(self.historical_data)
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using weighted moving average.
        
        Args:
            X: Feature matrix with columns ['store_id', 'product_id', 'dt']
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.debug(f"Making moving average predictions for {len(X)} samples")
        
        predictions = []
        new_items_count = 0
        
        for idx, row in X.iterrows():
            store_id = int(row['store_id'])
            product_id = int(row['product_id'])
            
            store_product_key = (store_id, product_id)
            
            if store_product_key in self.historical_data:
                window_stats = self.historical_data[store_product_key]
                
                # Combine multiple windows with exponential weighting
                pred = self._combine_windows(window_stats)
            else:
                # New store-product combination
                pred = self.default_value
                new_items_count += 1
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if new_items_count > 0:
            logger.warning(f"Found {new_items_count} new store-product combinations, "
                          f"using default value: {self.default_value}")
        
        return predictions
    
    def _combine_windows(self, window_stats: Dict) -> float:
        """
        Combine predictions from multiple windows using exponential weighting.
        
        Args:
            window_stats: Dictionary mapping window_size -> {mean, std, count}
            
        Returns:
            Combined weighted prediction
        """
        if not window_stats:
            return self.default_value
        
        windows = sorted(window_stats.keys())
        
        if self.weighting_method == 'exponential':
            # More recent (smaller) windows get higher weight
            # Use exponential decay: weight = alpha^(position from end)
            weights = []
            total_weight = 0
            
            for i, window in enumerate(windows):
                # Position from end: closer to end (recent) = higher weight
                position = len(windows) - i - 1
                weight = self.alpha ** position
                weights.append(weight)
                total_weight += weight
            
            # Normalize weights
            weights = np.array(weights) / total_weight
            
            # Compute weighted average
            prediction = 0
            for weight, window in zip(weights, windows):
                prediction += weight * window_stats[window]['mean']
            
            return float(prediction)
        
        elif self.weighting_method == 'equal':
            # Equal weight to all windows
            means = [window_stats[w]['mean'] for w in windows]
            return float(np.mean(means))
        
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of the model."""
        summary = super().get_model_summary()
        summary.update({
            'unique_items': len(self.historical_data),
            'windows': self.windows,
            'weighting_method': self.weighting_method,
            'alpha': self.alpha
        })
        return summary