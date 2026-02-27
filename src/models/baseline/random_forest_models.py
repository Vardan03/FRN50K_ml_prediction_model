# src/models/baseline/random_forest_models.py
"""
Random Forest regression baseline models for retail demand forecasting.

This module implements Random Forest Baselines with:
- Comprehensive feature engineering (temporal, cyclical, lag, rolling)
- Handling of high-dimensional and sparse data
- Non-negative prediction enforcement
- Feature importance analysis
- Full error handling and logging
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.models.base_model import BaseForecastingModel
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class RandomForestBaseline(BaseForecastingModel):
    """
    Random Forest Regression Baseline for retail demand forecasting.
    
    Key features:
    - Automatic temporal feature creation
    - Random Forest for non-linear relationships
    - Handles high-dimensional sparse data naturally
    - Feature importance scoring
    - Non-negative prediction enforcement
    - Comprehensive validation
    
    Random Forests are particularly well-suited for retail demand forecasting
    because they:
    - Handle non-linear relationships between features and target
    - Automatically detect feature interactions
    - Are robust to outliers
    - Can handle missing values effectively
    - Don't require feature scaling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Random Forest Baseline model.
        
        Args:
            config: Dictionary with configuration parameters:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: 20)
                - min_samples_split: Min samples for split (default: 5)
                - min_samples_leaf: Min samples per leaf (default: 2)
                - max_features: Features per split (default: 'sqrt')
                - n_jobs: Parallel jobs (default: -1, all CPUs)
                - random_state: Random seed (default: 42)
                - enforce_non_negative: Clip predictions to 0+ (default: True)
                - min_samples: Minimum training samples (default: 10)
                - lag_periods: Lag feature periods (default: [1, 7, 168])
                - rolling_windows: Rolling windows sizes (default: [6, 24, 168])
        """
        super().__init__(model_name="RandomForestRegression", config=config)
        
        # Default configuration for Random Forest
        self.config = config or {}
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 20)
        self.min_samples_split = self.config.get('min_samples_split', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 2)
        self.max_features = self.config.get('max_features', 'sqrt')
        self.n_jobs = self.config.get('n_jobs', -1)
        self.random_state = self.config.get('random_state', 42)
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.min_samples = self.config.get('min_samples', 10)
        
        # Feature engineering configuration
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Model components
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        self.feature_names = []
        self.feature_importance_dict = {}
        
        logger.info(
            f"RandomForestBaseline initialized: n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, max_features={self.max_features}"
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestBaseline':
        """
        Fit the Random Forest model with comprehensive preprocessing.
        
        Args:
            X: Feature dataframe with columns: store_id, product_id, dt, ...
            y: Target values (sale_amount)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If input validation fails
        """
        # Validate input
        self.validate_input(X, y)
        
        logger.info(f"Fitting RandomForestBaseline on {len(X)} samples")
        
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
        # Random Forest cannot handle NaN values directly
        logger.debug("Handling missing values...")
        
        # Forward fill first (using newer pandas API)
        X_features = X_features.ffill(limit=10)
        
        # Then backward fill for any remaining NaNs
        X_features = X_features.bfill(limit=10)
        
        # Fill remaining with 0
        X_features = X_features.fillna(0)
        
        # Replace infinite values with large but finite numbers
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Ensure no NaN or inf remain
        n_nans_before = X_features.isnull().sum().sum()
        
        if n_nans_before > 0:
            logger.warning(f"Remaining issues after initial cleaning: {n_nans_before} NaN")
            X_features = X_features.fillna(0)
            X_features[numeric_cols] = X_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Step 5: Handle sparse data - add features for missing patterns
        # Create binary flags for whether values were originally missing
        logger.debug("Creating missing value indicators...")
        X_missing_indicators = (X_engineered[self.feature_names].isnull()).astype(int)
        
        # Add indicators with prefix to avoid name conflicts
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
        
        # Step 6: Model training
        logger.debug("Training Random Forest model...")
        self.model.fit(X_features, y_train)
        
        # Step 7: Model diagnostics
        train_r2 = self.model.score(X_features, y_train)
        logger.info(f"Training R² score: {train_r2:.4f}")
        
        # Cross-validation score
        if len(X) >= 30:
            try:
                cv_scores = cross_val_score(
                    self.model, X_features, y_train,
                    cv=5, scoring='r2', n_jobs=self.n_jobs
                )
                logger.info(
                    f"Cross-validation R² scores: mean={cv_scores.mean():.4f}, "
                    f"std={cv_scores.std():.4f}"
                )
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Feature importance analysis
        logger.debug("Computing feature importance...")
        feature_importance = self.model.feature_importances_
        
        self.feature_importance_dict = dict(
            sorted(
                zip(self.feature_names, feature_importance),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        # Log top 10 important features
        top_10_importance = list(self.feature_importance_dict.items())[:10]
        logger.info(f"Top 10 important features:")
        for i, (feat_name, importance) in enumerate(top_10_importance, 1):
            logger.info(f"  {i}. {feat_name}: {importance:.4f}")
        
        # Mark as fitted
        self.is_fitted = True
        logger.info("Model training completed successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions with preprocessing and validation.
        
        Args:
            X: Feature dataframe with same structure as training data
            
        Returns:
            Predicted values (numpy array)
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.validate_input(X)
        logger.debug(f"Generating predictions for {len(X)} samples")
        
        # Step 1: Feature engineering (same pipeline as training)
        X_engineered = self.feature_engineer.engineer_all_features(
            X.copy(),
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        # Step 2: Extract same features as training
        # Get only the original feature names (without the indicator suffix)
        original_features = [f for f in self.feature_names 
                            if not f.startswith('is_missing_')]
        X_features = X_engineered[original_features].copy()
        
        # Step 3: Handle missing values
        logger.debug("Handling missing values in prediction data...")
        
        # Forward fill
        X_features = X_features.ffill(limit=10)
        
        # Backward fill
        X_features = X_features.bfill(limit=10)
        
        # Fill remaining with 0
        X_features = X_features.fillna(0)
        
        # Replace infinite values
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Step 4: Add missing value indicators (matching training)
        X_missing_indicators = (X_engineered[original_features].isnull()).astype(int)
        for col in X_missing_indicators.columns:
            indicator_col = f'is_missing_{col}'
            X_features[indicator_col] = X_missing_indicators[col]
        
        # Ensure feature order matches training
        X_features = X_features[self.feature_names]
        
        # Step 5: Generate predictions
        logger.debug("Running Random Forest prediction...")
        predictions = self.model.predict(X_features)
        
        # Step 6: Enforce non-negative constraint
        if self.enforce_non_negative:
            negative_count = (predictions < 0).sum()
            if negative_count > 0:
                logger.warning(
                    f"Found {negative_count} negative predictions, clipping to 0"
                )
            predictions = np.maximum(predictions, 0)
        
        logger.debug(f"Predictions generated: mean={predictions.mean():.4f}, "
                    f"std={predictions.std():.4f}, min={predictions.min():.4f}, "
                    f"max={predictions.max():.4f}")
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary with feature importance and diagnostics.
        
        Returns:
            Dictionary with model information
        """
        summary = super().get_model_summary()
        
        if self.is_fitted and self.model is not None:
            summary.update({
                'model_type': 'RandomForest',
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'n_features': len(self.feature_names),
                'n_trees': self.model.n_estimators,
                'top_10_important_features': dict(
                    list(self.feature_importance_dict.items())[:10]
                ) if self.feature_importance_dict else {},
                'total_feature_importance': float(np.sum(self.model.feature_importances_)),
            })
        
        return summary
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores for interpretability.
        
        Args:
            top_n: Number of top features to return (default: 20)
            
        Returns:
            Dictionary of feature names and importance scores
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        top_features = dict(list(self.feature_importance_dict.items())[:top_n])
        return top_features
    
    def get_feature_importance_all(self) -> Dict[str, float]:
        """Get feature importance for all features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.feature_importance_dict.copy()


class RandomForestAdvanced(BaseForecastingModel):
    """
    Advanced Random Forest with gradient boosting comparison.
    
    This is an extension that allows comparison between:
    - Standard Random Forest
    - Gradient Boosting (XGBoost if available)
    - Other ensemble methods
    
    Useful for benchmarking different algorithms on the same dataset.
    """
    
    def __init__(self, ensemble_type: str = 'random_forest', config: Optional[Dict] = None):
        """
        Initialize advanced Random Forest model with multiple ensemble options.
        
        Args:
            ensemble_type: Type of ensemble ('random_forest' or 'gradient_boosting')
            config: Model configuration dictionary
        """
        super().__init__(model_name=f"RandomForest_{ensemble_type}", config=config)
        
        self.ensemble_type = ensemble_type
        self.config = config or {}
        self.scaler = None
        self.feature_names = []
        
        # Feature engineering
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Initialize appropriate model
        if ensemble_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 20),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                n_jobs=-1,
                random_state=42,
            )
        elif ensemble_type == 'gradient_boosting':
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 5),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42,
                )
            except ImportError:
                logger.warning("GradientBoostingRegressor not available, using RandomForest")
                self.model = RandomForestRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 20),
                    n_jobs=-1,
                    random_state=42,
                )
        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}")
        
        logger.info(f"RandomForestAdvanced initialized with {ensemble_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestAdvanced':
        """Fit the ensemble model with feature engineering."""
        self.validate_input(X, y)
        
        logger.info(f"Fitting {self.ensemble_type} on {len(X)} samples")
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(X.copy())
        
        # Get feature names
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(
            X_engineered,
            exclude_cols=exclude_cols
        )
        
        # Extract features
        X_features = X_engineered[self.feature_names].copy()
        
        # Handle missing values
        X_features = X_features.fillna(X_features.mean())
        X_features = X_features.fillna(0)
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Train model
        self.model.fit(X_features, y)
        self.is_fitted = True
        
        # Log metrics
        train_r2 = self.model.score(X_features, y)
        logger.info(f"Training R² score: {train_r2:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions with preprocessing."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.validate_input(X)
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(X.copy())
        X_features = X_engineered[self.feature_names].copy()
        
        # Handle missing values
        X_features = X_features.fillna(0)
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Generate predictions
        predictions = self.model.predict(X_features)
        
        # Enforce non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        summary = super().get_model_summary()
        
        if self.is_fitted:
            summary.update({
                'ensemble_type': self.ensemble_type,
                'n_features': len(self.feature_names),
            })
            
            # Add feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(
                    self.feature_names[:10],
                    self.model.feature_importances_[:10]
                ))
                summary['top_10_features'] = importance
        
        return summary
