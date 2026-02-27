# src/models/baseline/linear_models.py
"""
Linear regression baseline models for retail demand forecasting.

This module implements Linear Regression baselines with:
- Comprehensive feature engineering (temporal, cyclical, lag, rolling)
- Feature scaling and normalization
- Non-negative prediction enforcement
- Sparse data handling
- Full error handling and logging
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.models.base_model import BaseForecastingModel
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class LinearRegressionBaseline(BaseForecastingModel):
    """
    Linear Regression Baseline for retail demand forecasting.
    
    Key features:
    - Automatic temporal feature creation
    - Feature scaling for numerical stability
    - Ridge regression to handle multicollinearity
    - Non-negative prediction enforcement
    - Comprehensive validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Linear Regression Baseline model."""
        super().__init__(model_name="LinearRegression", config=config)
        
        # Default configuration
        self.config = config or {}
        self.alpha = self.config.get('alpha', 1.0)
        self.use_ridge = self.config.get('use_ridge', True)
        self.scaler_type = self.config.get('scaler_type', 'standard')
        self.min_samples = self.config.get('min_samples', 10)
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        
        # Feature engineering configuration
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if self.scaler_type == 'standard' else None
        self.feature_names = []
        self.feature_means = {}
        self.feature_stds = {}
        
        logger.info(
            f"LinearRegressionBaseline initialized: alpha={self.alpha}, "
            f"use_ridge={self.use_ridge}, scaler={self.scaler_type}"
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LinearRegressionBaseline':
        """
        Fit the linear regression model with comprehensive preprocessing.
        
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
        
        logger.info(f"Fitting LinearRegressionBaseline on {len(X)} samples")
        
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
        
        # Step 4: Handle any remaining NaN values in features
        X_features = X_features.fillna(X_features.mean())
        X_features = X_features.fillna(0)  # Fill any remaining NaN with 0
        
        # Replace inf values
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Store statistics for prediction preprocessing
        self.feature_means = X_features.mean().to_dict()
        self.feature_stds = X_features.std().to_dict()
        
        # Step 5: Feature scaling
        logger.debug("Scaling features...")
        if self.scaler is not None:
            X_features_scaled = self.scaler.fit_transform(X_features)
        else:
            X_features_scaled = X_features.values
        
        # Step 6: Model selection and training
        logger.debug(f"Training {'Ridge' if self.use_ridge else 'Linear'} regression...")
        
        if self.use_ridge:
            self.model = Ridge(alpha=self.alpha, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_features_scaled, y_train)
        
        # Step 7: Model diagnostics
        train_r2 = self.model.score(X_features_scaled, y_train)
        logger.info(f"Training R² score: {train_r2:.4f}")
        
        # Cross-validation score
        if len(X) >= 30:
            cv_scores = cross_val_score(
                self.model, X_features_scaled, y_train,
                cv=5, scoring='r2'
            )
            logger.info(
                f"Cross-validation R² scores: mean={cv_scores.mean():.4f}, "
                f"std={cv_scores.std():.4f}"
            )
        
        # Feature importance (coefficients)
        if len(self.feature_names) <= 20:
            top_features = sorted(
                zip(self.feature_names, np.abs(self.model.coef_)),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            logger.info(f"Top 5 important features: {top_features}")
        
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
        X_features = X_engineered[self.feature_names].copy()
        
        # Step 3: Handle missing values using training statistics
        for col in X_features.columns:
            if X_features[col].isnull().any():
                X_features[col] = X_features[col].fillna(
                    self.feature_means.get(col, X_features[col].mean())
                )
        
        # Fill remaining NaN and inf
        X_features = X_features.fillna(0)
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Step 4: Scale features using training scaler
        if self.scaler is not None:
            X_features_scaled = self.scaler.transform(X_features)
        else:
            X_features_scaled = X_features.values
        
        # Step 5: Generate predictions
        predictions = self.model.predict(X_features_scaled)
        
        # Step 6: Enforce non-negative constraint
        if self.enforce_non_negative:
            negative_count = (predictions < 0).sum()
            if negative_count > 0:
                logger.warning(
                    f"Found {negative_count} negative predictions, clipping to 0"
                )
            predictions = np.maximum(predictions, 0)
        
        logger.debug(f"Predictions generated: mean={predictions.mean():.4f}, "
                    f"std={predictions.std():.4f}")
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary with coefficients and diagnostics."""
        summary = super().get_model_summary()
        
        if self.is_fitted and self.model is not None:
            summary.update({
                'model_type': 'Ridge' if self.use_ridge else 'Linear',
                'alpha': self.alpha if self.use_ridge else None,
                'n_features': len(self.feature_names),
                'coefficients': dict(zip(
                    self.feature_names[:10],
                    self.model.coef_[:10]
                )) if hasattr(self.model, 'coef_') else {},
                'intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None,
            })
        
        return summary


class LinearForecastingModel(BaseForecastingModel):
    """Legacy Linear Forecasting Model (kept for compatibility)."""
    
    def __init__(self, model_type: str = 'linear', config: dict = None):
        """Initialize linear model with specified type and configuration."""
        super().__init__(f"LinearRegression_{model_type}", config)
        self.model_type = model_type
        self.scaler = None
        
        # Configuration with sensible defaults
        self.use_scaling = config.get('use_scaling', True) if config else True
        self.handle_missing = config.get('handle_missing', True) if config else True
        
        # Initialize the appropriate sklearn model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            alpha = config.get('alpha', 1.0) if config else 1.0
            self.model = Ridge(alpha=alpha, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Prepare features for modeling: handle missing values, scale, etc."""
        X_processed = X.copy()
        
        # Handle missing values
        if self.handle_missing:
            X_processed = X_processed.fillna(X_processed.mean())
        
        # Feature scaling
        if self.use_scaling:
            if is_training:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted")
                X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LinearForecastingModel':
        """Fit the linear model with comprehensive preprocessing."""
        self.validate_input(X, y)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Remove non-numeric columns for linear regression
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(numeric_cols) < len(X.columns):
            dropped_cols = set(X.columns) - set(numeric_cols)
            logger.warning(f"Dropped non-numeric columns: {list(dropped_cols)}")
            self.feature_names = numeric_cols.tolist()
        
        logger.info(f"Fitting {self.model_name} with {len(self.feature_names)} features...")
        
        # Prepare features
        X_processed = self._prepare_features(X_numeric, is_training=True)
        
        # Handle any remaining infinite values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Fit the model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Store training metrics for analysis
        train_score = self.model.score(X_processed, y)
        self.training_history['train_r2'] = train_score
        self.training_history['n_features'] = X_processed.shape[1]
        self.training_history['n_samples'] = X_processed.shape[0]
        
        logger.info(f"Training completed. R² score: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions with the same preprocessing as training."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use only the features that were used during training
        X_features = X[self.feature_names]
        
        # Apply the same preprocessing pipeline
        X_processed = self._prepare_features(X_features, is_training=False)
        
        # Handle infinite values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Generate predictions
        predictions = self.model.predict(X_processed)
        
        # Ensure non-negative predictions for demand forecasting
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients for interpretability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        
        if hasattr(self.model, 'coef_'):
            coefficients = dict(zip(self.feature_names, self.model.coef_))
            
            # Add intercept if available
            if hasattr(self.model, 'intercept_'):
                coefficients['intercept'] = self.model.intercept_
            
            return coefficients
        else:
            return {}
