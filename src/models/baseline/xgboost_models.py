# src/models/baseline/xgboost_models.py
"""
XGBoost regression baseline models for retail demand forecasting.

This module implements XGBoost Baselines with:
- Comprehensive feature engineering (temporal, cyclical, lag, rolling)
- Gradient boosting for high-dimensional data
- Handling of sparse data and missing values
- Non-negative prediction enforcement
- Feature importance analysis
- Full error handling and logging

XGBoost is particularly well-suited for retail demand forecasting because it:
- Handles non-linear relationships with boosting iterations
- Manages feature interactions through tree-based learning
- Provides native missing value handling
- Offers feature importance through multiple metrics
- Scales well with high-dimensional feature sets
- Provides regularization to prevent overfitting
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from src.models.base_model import BaseForecastingModel
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class XGBoostBaseline(BaseForecastingModel):
    """
    XGBoost Regression Baseline for retail demand forecasting.
    
    Key features:
    - Automatic temporal feature creation via FeatureEngineer
    - XGBoost gradient boosting for complex patterns
    - Native handling of missing values and sparse data
    - Feature importance scoring (gain, cover, frequency)
    - Non-negative prediction enforcement
    - Comprehensive validation and error handling
    
    Configuration parameters:
    - n_estimators: Number of boosting rounds (default: 100)
    - max_depth: Maximum tree depth (default: 6)
    - learning_rate: Learning rate/eta (default: 0.1)
    - subsample: Row sampling ratio (default: 0.8)
    - colsample_bytree: Feature sampling ratio (default: 0.8)
    - colsample_bylevel: Feature sampling per level (default: 0.8)
    - reg_alpha: L1 regularization (default: 0.0)
    - reg_lambda: L2 regularization (default: 1.0)
    - gamma: Minimum loss reduction (default: 0.0)
    - min_child_weight: Minimum child weight (default: 1)
    - tree_method: Tree construction algorithm (default: 'auto')
    - objective: Loss function (default: 'reg:squarederror')
    - n_jobs: Parallel jobs (default: -1)
    - random_state: Random seed (default: 42)
    - enforce_non_negative: Clip predictions to 0+ (default: True)
    - min_samples: Minimum training samples (default: 10)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize XGBoost Baseline model.
        
        Args:
            config: Dictionary with XGBoost hyperparameters and model settings
            
        Raises:
            ImportError: If xgboost is not installed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        super().__init__(model_name="XGBoostRegression", config=config)
        
        # Default configuration for XGBoost
        self.config = config or {}
        
        # XGBoost hyperparameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
        self.colsample_bylevel = self.config.get('colsample_bylevel', 0.8)
        self.reg_alpha = self.config.get('reg_alpha', 0.0)
        self.reg_lambda = self.config.get('reg_lambda', 1.0)
        self.gamma = self.config.get('gamma', 0.0)
        self.min_child_weight = self.config.get('min_child_weight', 1)
        self.tree_method = self.config.get('tree_method', 'auto')
        self.objective = self.config.get('objective', 'reg:squarederror')
        self.n_jobs = self.config.get('n_jobs', -1)
        self.random_state = self.config.get('random_state', 42)
        
        # Model-specific settings
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.min_samples = self.config.get('min_samples', 10)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 10)
        self.verbose_eval = self.config.get('verbose_eval', 0)
        
        # Feature engineering configuration
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            objective=self.objective,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0
        )
        
        # Model state
        self.feature_names = []
        self.feature_importance_dict = {}
        self.training_history = {}
        
        logger.info(
            f"XGBoostBaseline initialized: n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, learning_rate={self.learning_rate}, "
            f"reg_alpha={self.reg_alpha}, reg_lambda={self.reg_lambda}"
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'XGBoostBaseline':
        """
        Fit the XGBoost model with comprehensive preprocessing.
        
        Args:
            X: Feature dataframe with columns: store_id, product_id, dt, ...
            y: Target values (sale_amount)
            X_val: Optional validation set for early stopping
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If training fails
        """
        # Validate input
        self.validate_input(X, y)
        
        logger.info(f"Fitting XGBoostBaseline on {len(X)} samples")
        
        if len(X) < self.min_samples:
            raise ValueError(
                f"Insufficient samples for training: {len(X)} < {self.min_samples}"
            )
        
        # Step 1: Feature Engineering
        logger.debug("Starting feature engineering pipeline...")
        X_engineered = self.feature_engineer.engineer_all_features(
            X.copy(),
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        # Step 2: Get feature names (exclude identifiers)
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(
            X_engineered,
            exclude_cols=exclude_cols
        )
        
        logger.info(f"Using {len(self.feature_names)} features for training")
        
        # Step 3: Extract features and target
        X_features = X_engineered[self.feature_names].copy()
        y_train = y.copy()
        
        # Step 4: Handle NaN and inf values
        logger.debug("Handling missing values...")
        
        # Forward fill first
        X_features = X_features.ffill(limit=10)
        
        # Backward fill for any remaining NaNs
        X_features = X_features.bfill(limit=10)
        
        # Fill remaining with 0
        X_features = X_features.fillna(0)
        
        # Replace infinite values
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Step 5: Handle sparse data - add missing value indicators
        logger.debug("Creating missing value indicators...")
        X_missing_indicators = (X_engineered[self.feature_names].isnull()).astype(int)
        
        for col in X_missing_indicators.columns:
            indicator_col = f'is_missing_{col}'
            X_features[indicator_col] = X_missing_indicators[col]
        
        # Update feature names
        new_features = [col for col in X_features.columns if col not in self.feature_names]
        self.feature_names = self.feature_names + new_features
        
        logger.info(
            f"After adding missing indicators: {len(self.feature_names)} features "
            f"({len(new_features)} indicators added)"
        )
        
        # Step 6: Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            logger.debug("Processing validation set...")
            
            X_val_engineered = self.feature_engineer.engineer_all_features(
                X_val.copy(),
                handle_missing=True,
                missing_strategy='forward_fill'
            )
            
            X_val_features = X_val_engineered[self.feature_names[:len(self.feature_names)-len(new_features)]].copy()
            
            # Apply same preprocessing
            X_val_features = X_val_features.ffill(limit=10)
            X_val_features = X_val_features.bfill(limit=10)
            X_val_features = X_val_features.fillna(0)
            X_val_features[numeric_cols] = X_val_features[numeric_cols].replace([np.inf, -np.inf], 0)
            
            # Add missing indicators
            X_val_missing = (X_val_engineered[self.feature_names[:len(self.feature_names)-len(new_features)]].isnull()).astype(int)
            for col in X_val_missing.columns:
                indicator_col = f'is_missing_{col}'
                X_val_features[indicator_col] = X_val_missing[col]
            
            eval_set = [(X_val_features.values, y_val.values)]
        
        # Step 7: Model training
        logger.debug("Training XGBoost model...")
        
        try:
            train_params = {
                'eval_set': eval_set,
                'verbose': False,
                'sample_weight': None
            }
            
            if eval_set is not None:
                train_params['early_stopping_rounds'] = self.early_stopping_rounds
            
            self.model.fit(X_features.values, y_train.values, **train_params)
            
            # Store feature names for prediction
            self._model_feature_count = X_features.shape[1]
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"XGBoost training failed: {str(e)}")
        
        # Step 8: Model diagnostics
        train_r2 = self.model.score(X_features.values, y_train.values)
        logger.info(f"Training R² score: {train_r2:.4f}")
        
        # Cross-validation score
        if len(X) >= 30:
            try:
                cv_scores = cross_val_score(
                    xgb.XGBRegressor(
                        n_estimators=50,  # Use fewer estimators for CV
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=0
                    ),
                    X_features.values,
                    y_train.values,
                    cv=5,
                    scoring='r2',
                    n_jobs=self.n_jobs
                )
                logger.info(
                    f"Cross-validation R² scores: {cv_scores.mean():.4f} "
                    f"(+/- {cv_scores.std():.4f})"
                )
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Extract feature importance
        self._extract_feature_importance()
        
        logger.info(f"Model training completed successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data with preprocessing consistency.
        
        Args:
            X: Feature dataframe with same structure as training data
            
        Returns:
            Array of predictions
            
        Raises:
            RuntimeError: If model is not fitted
            ValueError: If input validation fails
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        logger.debug(f"Making predictions on {len(X)} samples")
        
        # Apply same preprocessing as training
        X_engineered = self.feature_engineer.engineer_all_features(
            X.copy(),
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        feature_names = FeatureEngineer.get_feature_names(
            X_engineered,
            exclude_cols=exclude_cols
        )
        
        X_features = X_engineered[feature_names].copy()
        
        # Handle missing values
        X_features = X_features.ffill(limit=10)
        X_features = X_features.bfill(limit=10)
        X_features = X_features.fillna(0)
        
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Add missing indicators
        X_missing_indicators = (X_engineered[feature_names].isnull()).astype(int)
        for col in X_missing_indicators.columns:
            indicator_col = f'is_missing_{col}'
            X_features[indicator_col] = X_missing_indicators[col]
        
        # Ensure feature count matches training
        if X_features.shape[1] != self._model_feature_count:
            logger.warning(
                f"Feature count mismatch: {X_features.shape[1]} != "
                f"{self._model_feature_count}. Attempting to reconcile..."
            )
            
            if X_features.shape[1] < self._model_feature_count:
                # Add missing features as zeros
                missing_count = self._model_feature_count - X_features.shape[1]
                for i in range(missing_count):
                    X_features[f'missing_feature_{i}'] = 0
            else:
                # Remove extra features
                X_features = X_features.iloc[:, :self._model_feature_count]
        
        # Make predictions
        predictions = self.model.predict(X_features.values)
        
        # Enforce non-negative predictions
        if self.enforce_non_negative:
            n_clipped = np.sum(predictions < 0)
            if n_clipped > 0:
                logger.debug(f"Clipping {n_clipped} negative predictions to 0")
                predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from trained model."""
        try:
            # Get feature importance from the model
            importance_dict = self.model.get_booster().get_score(
                importance_type='weight'
            )
            
            # Convert to DataFrame for better analysis
            if importance_dict:
                total_importance = sum(importance_dict.values())
                self.feature_importance_dict = {
                    k: (v / total_importance * 100) 
                    for k, v in sorted(
                        importance_dict.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                }
                
                logger.debug(
                    f"Extracted importance for {len(self.feature_importance_dict)} features"
                )
            else:
                logger.warning("No feature importance information available")
                self.feature_importance_dict = {}
                
        except Exception as e:
            logger.warning(f"Error extracting feature importance: {str(e)}")
            self.feature_importance_dict = {}
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.feature_importance_dict:
            logger.warning("No feature importance available")
            return pd.DataFrame({'feature': [], 'importance': []})
        
        # Get top N features
        top_features = dict(sorted(
            self.feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        df = pd.DataFrame({
            'feature': list(top_features.keys()),
            'importance': list(top_features.values())
        })
        
        df['cumulative_importance'] = df['importance'].cumsum()
        
        return df
    
    def get_feature_importance_all(self) -> pd.DataFrame:
        """
        Get all features sorted by importance.
        
        Returns:
            DataFrame with all features and importance scores
        """
        if not self.feature_importance_dict:
            logger.warning("No feature importance available")
            return pd.DataFrame({'feature': [], 'importance': []})
        
        df = pd.DataFrame({
            'feature': list(self.feature_importance_dict.keys()),
            'importance': list(self.feature_importance_dict.values())
        })
        
        df['cumulative_importance'] = df['importance'].cumsum()
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary and diagnostics.
        
        Returns:
            Dictionary with model statistics
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted first")
        
        return {
            'model_name': self.model_name,
            'n_features': len(self.feature_names),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'feature_names': self.feature_names[:20],  # First 20 features
            'total_features_used': len(self.feature_names),
            'enforce_non_negative': self.enforce_non_negative,
            'top_features': self.get_feature_importance(top_n=10).to_dict('records'),
        }


class XGBoostAdvanced(BaseForecastingModel):
    """
    Advanced XGBoost with ensemble methods and custom configurations.
    
    Provides:
    - Flexible algorithm selection (XGBoost, LightGBM)
    - Custom hyperparameter configurations
    - Cross-validation capabilities
    - Ensemble comparison utilities
    """
    
    def __init__(self, algorithm: str = 'xgboost', config: Optional[Dict] = None):
        """
        Initialize Advanced model.
        
        Args:
            algorithm: 'xgboost' or 'lightgbm' (default: 'xgboost')
            config: Configuration dictionary
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        super().__init__(model_name=f"XGBoostAdvanced_{algorithm}", config=config)
        
        if algorithm not in ['xgboost']:
            raise ValueError(f"Algorithm {algorithm} not supported. Use 'xgboost'")
        
        self.algorithm = algorithm
        self.config = config or {}
        
        # Initialize appropriate model
        if algorithm == 'xgboost':
            self.model = XGBoostBaseline(config=config)
        
        logger.info(f"XGBoostAdvanced initialized with {algorithm}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostAdvanced':
        """Fit the model."""
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return self.model.get_model_summary()
