import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ForecastingMetrics:
    """
    Comprehensive evaluation metrics for demand forecasting models.
    
    Teaching purpose: Understanding how to properly evaluate forecasting models
    is crucial. Different metrics capture different aspects of model performance.
    """
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error - average absolute difference between predictions and actual values.
        
        Teaching insight: MAE is easy to interpret (same units as target variable)
        and robust to outliers. Good for understanding typical prediction errors.
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error - penalizes large errors more than small ones.
        
        Teaching insight: RMSE is more sensitive to outliers than MAE.
        Use when large errors are particularly problematic for your business.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Mean Absolute Percentage Error - percentage-based error metric.
        
        Teaching insight: MAPE is scale-independent, making it good for comparing
        across different products or stores. However, it can be problematic when
        actual values are close to zero (division by zero issue).
        """
        # Add small epsilon to avoid division by zero
        denominator = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Symmetric MAPE - addresses some issues with traditional MAPE.
        
        Teaching insight: sMAPE treats over-forecasting and under-forecasting
        more symmetrically than regular MAPE.
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Bias - measures systematic over or under-prediction.
        
        Teaching insight: Bias is crucial in retail forecasting. Consistent
        over-forecasting leads to excess inventory; under-forecasting leads to stockouts.
        """
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def bias_percentage(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Bias as a percentage of actual values."""
        mean_actual = np.mean(y_true) + epsilon
        return (np.mean(y_pred - y_true) / mean_actual) * 100
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R² Score (Coefficient of Determination).
        
        Teaching insight: R² measures the proportion of variance in the target
        variable that is explained by the model. Range: (-∞, 1]
        - 1.0 = perfect predictions
        - 0.0 = model no better than predicting mean
        - <0 = model worse than naive mean forecast
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return np.nan
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1) -> float:
        """
        Mean Absolute Scaled Error - compares model performance to naive forecast.
        
        Teaching insight: MASE is scale-independent and compares your model
        to a simple baseline. Values < 1 mean your model beats the naive forecast.
        """
        if len(y_train) <= seasonal_period:
            # Fallback to simple naive forecast if insufficient training data
            naive_forecast = np.full_like(y_true, np.mean(y_train))
            mae_naive = ForecastingMetrics.mean_absolute_error(y_true, naive_forecast)
        else:
            # Use seasonal naive forecast
            mae_naive = np.mean(np.abs(np.diff(y_train[-seasonal_period:])))
        
        mae_model = ForecastingMetrics.mean_absolute_error(y_true, y_pred)
        
        return mae_model / (mae_naive + 1e-8)  # Avoid division by zero
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics at once.
        
        This is the main function students will use to evaluate their models.
        """
        metrics = {
            'MAE': ForecastingMetrics.mean_absolute_error(y_true, y_pred),
            'RMSE': ForecastingMetrics.root_mean_squared_error(y_true, y_pred),
            'MAPE': ForecastingMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'sMAPE': ForecastingMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'Bias': ForecastingMetrics.bias(y_true, y_pred),
            'Bias_Percentage': ForecastingMetrics.bias_percentage(y_true, y_pred),
            'R²': ForecastingMetrics.r_squared(y_true, y_pred)
        }
        
        # Add MASE if training data is available
        if y_train is not None:
            metrics['MASE'] = ForecastingMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        return metrics
    
    @staticmethod
    def evaluate_by_group(df: pd.DataFrame, y_true_col: str, y_pred_col: str, 
                         group_cols: List[str], y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate model performance by different groupings (store, product, etc.).
        
        Teaching value: Understanding where your model performs well or poorly
        is crucial for model improvement and business insights.
        """
        results = []
        
        for group_values, group_data in df.groupby(group_cols):
            y_true_group = group_data[y_true_col].values
            y_pred_group = group_data[y_pred_col].values
            
            # Calculate metrics for this group
            group_metrics = ForecastingMetrics.calculate_all_metrics(y_true_group, y_pred_group, y_train)
            
            # Add group identifiers
            group_result = dict(zip(group_cols, group_values if isinstance(group_values, tuple) else [group_values]))
            group_result.update(group_metrics)
            group_result['n_observations'] = len(group_data)
            
            results.append(group_result)
        
        return pd.DataFrame(results)