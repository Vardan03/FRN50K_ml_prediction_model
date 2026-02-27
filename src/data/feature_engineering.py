import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for retail demand forecasting.
    
    This class teaches students the art and science of feature engineering,
    which is often more impactful than choosing sophisticated algorithms.
    """
    
    def __init__(self, config: dict):
        """
        Initialize with preprocessing configuration.
        
        Teaching point: Feature engineering parameters should be configurable
        to enable easy experimentation and hyperparameter tuning.
        """
        self.config = config
        self.lag_periods = config['lag_periods']
        self.rolling_windows = config['rolling_windows']
        self.seasonal_periods = config['seasonal_periods']
        
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'dt') -> pd.DataFrame:
        """
        Create comprehensive time-based features.
        
        Teaching insight: Time-based features are crucial for retail forecasting
        because demand patterns are highly temporal (hourly, daily, weekly cycles).
        """
        df = df.copy()
        
        # Ensure datetime column is properly formatted
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        logger.info("Creating temporal features...")
        
        # Basic temporal components
        # These capture different levels of seasonality
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['quarter'] = df[datetime_col].dt.quarter
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        
        # Cyclical encoding - This is a key teaching moment!
        # Linear encoding (hour=23, hour=0) suggests these times are very different
        # Cyclical encoding correctly represents that 23:00 and 01:00 are close
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business-relevant time indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        
        # Retail-specific time periods (teaching: domain knowledge matters!)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Time since start of dataset (trend component)
        df['days_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.days
        df['hours_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.total_seconds() / 3600
        
        logger.info(f"Created {len([c for c in df.columns if c not in df.columns[:20]])} temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create lagged versions of the target variable.
        
        Teaching concept: Lag features capture the autocorrelation in time series.
        They're essential because past sales often predict future sales.
        """
        df = df.copy()
        
        # Sort data properly for lag calculation
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
        logger.info(f"Creating lag features for periods: {self.lag_periods}")
        
        # Create lag features for each store-product combination
        for lag in self.lag_periods:
            lag_col = f'{target_col}_lag_{lag}'
            
            # Use groupby to ensure lags are calculated within each time series
            df[lag_col] = df.groupby(['store_id', 'product_id'])[target_col].shift(lag)
            
            # For educational purposes, let's also create lag features for stockout status
            if lag <= 7:  # Only short-term lags for stockout (memory constraints)
                stockout_lag_col = f'stockout_lag_{lag}'
                df[stockout_lag_col] = df.groupby(['store_id', 'product_id'])['hours_stock_status'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Teaching insight: Rolling features capture local trends and patterns.
        They help the model understand if demand is increasing, decreasing, or stable.
        """
        df = df.copy()
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
        logger.info(f"Creating rolling features for windows: {self.rolling_windows}")
        
        for window in self.rolling_windows:
            # Rolling mean (local average demand)
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling standard deviation (demand volatility)
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)  # Fill NaN with 0 for single observations
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling maximum (peak demand in window)
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling median (robust central tendency)
            df[f'{target_col}_rolling_median_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from categorical variables.
        
        Teaching point: Categorical features often contain rich information
        that needs to be extracted thoughtfully.
        """
        df = df.copy()
        
        logger.info("Creating categorical and interaction features...")
        
        # Store-level aggregations (store characteristics)
        store_stats = df.groupby('store_id').agg({
            'sale_amount': ['mean', 'std', 'max'],
            'product_id': 'nunique'  # Number of products per store
        }).round(2)
        
        store_stats.columns = ['store_avg_sales', 'store_sales_volatility', 'store_max_sales', 'store_product_count']
        df = df.merge(store_stats, on='store_id', how='left')
        
        # Product-level aggregations (product characteristics)
        product_stats = df.groupby('product_id').agg({
            'sale_amount': ['mean', 'std'],
            'store_id': 'nunique'  # Number of stores selling this product
        }).round(2)
        
        product_stats.columns = ['product_avg_sales', 'product_sales_volatility', 'product_store_count']
        df = df.merge(product_stats, on='product_id', how='left')
        
        # City-level features (regional effects)
        city_stats = df.groupby('city_id').agg({
            'sale_amount': 'mean',
            'hours_stock_status': 'mean'  # Average stock availability by city
        }).round(2)
        
        city_stats.columns = ['city_avg_sales', 'city_stock_availability']
        df = df.merge(city_stats, on='city_id', how='left')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Teaching concept: Sometimes the combination of features is more
        informative than individual features alone.
        """
        df = df.copy()
        
        logger.info("Creating interaction features...")
        
        # Weather-time interactions (weather effect varies by time)
        df['temp_hour_interaction'] = df['avg_temperature'] * df['hour']
        df['rain_weekend_interaction'] = df['precpt'] * df['is_weekend']
        
        # Promotion-time interactions (promotion effectiveness varies by time)
        df['discount_weekend'] = df['discount'] * df['is_weekend']
        df['discount_evening'] = df['discount'] * df['is_evening_rush']
        
        # Stock-time interactions
        df['stock_hour_interaction'] = df['hours_stock_status'] * df['hour']
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values appropriately for time series data.
        
        Teaching concept: Missing value handling affects model performance.
        For time series, forward fill is often preferred over mean imputation.
        
        Args:
            df: Input dataframe
            strategy: 'forward_fill', 'backward_fill', 'mean', or 'zero'
        
        Returns:
            Dataframe with missing values handled
        """
        df = df.copy()
        
        # Identify missing columns
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("No missing values detected")
            return df
        
        logger.info(f"Handling {len(missing_cols)} columns with missing values using {strategy} strategy")
        
        if strategy == 'forward_fill':
            # Group by store-product and forward fill within groups
            df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
            for col in missing_cols:
                df[col] = df.groupby(['store_id', 'product_id'])[col].fillna(method='ffill')
                # Backward fill for values at start of time series
                df[col] = df[col].fillna(method='bfill')
                # Fill any remaining with mean
                df[col] = df[col].fillna(df[col].mean())
        
        elif strategy == 'backward_fill':
            for col in missing_cols:
                df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
                df[col] = df.groupby(['store_id', 'product_id'])[col].fillna(method='bfill')
                df[col] = df[col].fillna(method='ffill')
                df[col] = df[col].fillna(df[col].mean())
        
        elif strategy == 'mean':
            for col in missing_cols:
                df[col] = df[col].fillna(df[col].mean())
        
        elif strategy == 'zero':
            for col in missing_cols:
                df[col] = df[col].fillna(0)
        
        logger.info(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, handle_missing: bool = True, missing_strategy: str = 'forward_fill', skip_categorical: bool = False) -> pd.DataFrame:
        """
        Apply all feature engineering steps in the correct order.
        
        Teaching point: Feature engineering pipeline order matters!
        Some features depend on others being created first.
        
        Args:
            df: Input dataframe
            handle_missing: Whether to handle missing values
            missing_strategy: Strategy for handling missing values
            skip_categorical: Whether to skip categorical feature creation (useful when computing aggregations separately)
        
        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        original_cols = len(df.columns)
        
        # Step 0: Handle missing values if needed
        if handle_missing:
            df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 1: Temporal features (these don't depend on other features)
        df = self.create_temporal_features(df)
        
        # Step 2: Categorical features (these create new base features)
        if not skip_categorical:
            try:
                df = self.create_categorical_features(df)
            except Exception as e:
                logger.warning(f"Categorical features failed (expected for synthetic data): {e}")
        
        # Step 3: Lag features (these depend on having clean temporal data)
        try:
            df = self.create_lag_features(df)
        except Exception as e:
            logger.warning(f"Lag features skipped: {e}")
        
        # Step 4: Rolling features (these depend on temporal ordering)
        try:
            df = self.create_rolling_features(df)
        except Exception as e:
            logger.warning(f"Rolling features skipped: {e}")
        
        # Step 5: Interaction features (these depend on base features existing)
        df = self.create_interaction_features(df)
        
        final_cols = len(df.columns)
        logger.info(f"Feature engineering complete: {original_cols} → {final_cols} features (+{final_cols - original_cols})")
        
        # Fill any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    @staticmethod
    def get_feature_names(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature column names (excluding identifiers and target).
        
        Teaching: Feature selection is crucial for model performance and interpretability.
        """
        if exclude_cols is None:
            exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols