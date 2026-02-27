# src/models/baseline/ensemble_models.py
"""
Ensemble regression baseline models for retail demand forecasting.

This module implements advanced ensemble methods combining:
- Stacking (multiple learners + meta-learner)
- Voting (weighted averaging of predictions)
- Blending (holdout set for meta-learner)
- Bagging variants (bootstrap aggregation)
- Custom ensemble with optimal weighting

Supports:
- Comprehensive feature engineering
- High-dimensional and sparse data handling
- Non-negative prediction enforcement
- Cross-validation and diagnostics
- Full error handling and logging
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.models.base_model import BaseForecastingModel
from src.data.feature_engineering import FeatureEngineer
from src.models.baseline.linear_models import LinearRegressionBaseline
from src.models.baseline.random_forest_models import RandomForestBaseline
from src.models.baseline.naive_models import SeasonalNaiveModel

logger = logging.getLogger(__name__)


class StackingEnsembleRegressor(BaseForecastingModel):
    """
    Stacking Ensemble for retail demand forecasting.
    
    Stacking combines multiple base learners' predictions using a meta-learner:
    1. Train base learners on training data
    2. Generate meta-features using cross-validation
    3. Train meta-learner on meta-features
    4. Final predictions use base learners → meta-learner
    
    Benefits:
    - Combines strengths of multiple algorithms
    - Meta-learner learns optimal combination
    - Better generalization than single models
    - Handles diverse feature relationships
    
    Configuration:
    - base_learners: List of base model configs
    - meta_learner_type: 'linear', 'ridge', 'rf', 'xgb'
    - cv_splits: Cross-validation folds (default: 5)
    - random_state: Reproducibility seed (default: 42)
    - enforce_non_negative: Clip predictions to 0+
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Stacking Ensemble model.
        
        Args:
            config: Configuration dictionary with ensemble parameters
        """
        super().__init__(model_name="StackingEnsemble", config=config)
        
        self.config = config or {}
        self.cv_splits = self.config.get('cv_splits', 5)
        self.meta_learner_type = self.config.get('meta_learner_type', 'ridge')
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.random_state = self.config.get('random_state', 42)
        self.min_samples = self.config.get('min_samples', 10)
        
        # Feature engineering
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Initialize base learners
        self.base_learners = []
        self._init_base_learners()
        
        # Meta learner
        self.meta_learner = None
        self.scaler = StandardScaler()
        
        # State
        self.feature_names = []
        self.base_predictions_train = None
        self.meta_features_shape = None
        
        logger.info(
            f"StackingEnsembleRegressor initialized: meta_learner={self.meta_learner_type}, "
            f"cv_splits={self.cv_splits}, n_base_learners={len(self.base_learners)}"
        )
    
    def _init_base_learners(self):
        """Initialize base learners for stacking."""
        self.base_learners = [
            LinearRegressionBaseline(config={'alpha': 1.0, 'enforce_non_negative': False}),
            RandomForestBaseline(config={'n_estimators': 50, 'max_depth': 6, 'enforce_non_negative': False}),
            SeasonalNaiveModel(config={'seasonal_period': 168, 'fallback_method': 'simple_naive'}),
        ]
        
        if XGBOOST_AVAILABLE:
            try:
                from src.models.baseline.xgboost_models import XGBoostBaseline
                self.base_learners.append(
                    XGBoostBaseline(config={'n_estimators': 50, 'max_depth': 4, 'enforce_non_negative': False})
                )
            except Exception as e:
                logger.warning(f"Could not add XGBoost to base learners: {e}")
        
        logger.info(f"Initialized {len(self.base_learners)} base learners")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'StackingEnsembleRegressor':
        """
        Fit stacking ensemble with cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target values
            
        Returns:
            self for method chaining
        """
        self.validate_input(X, y)
        
        logger.info(f"Fitting StackingEnsembleRegressor on {len(X)} samples")
        
        if len(X) < self.min_samples * 2:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples * 2}")
        
        # Feature engineering
        logger.debug("Starting feature engineering...")
        X_engineered = self.feature_engineer.engineer_all_features(
            X.copy(),
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        # Get feature names
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(
            X_engineered,
            exclude_cols=exclude_cols
        )
        
        X_features = X_engineered[self.feature_names].copy()
        X_features = self._preprocess_features(X_features)
        y_train = y.copy()
        
        # Step 1: Generate meta-features using cross-validation
        logger.debug("Generating meta-features via cross-validation...")
        meta_features = np.zeros((len(X), len(self.base_learners)))
        
        for fold, (train_idx, val_idx) in enumerate(self._get_cv_splits(len(X))):
            X_fold_train = X_features.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_features.iloc[val_idx]
            
            for i, base_learner in enumerate(self.base_learners):
                try:
                    base_learner.fit(X.iloc[train_idx], y_fold_train)
                    meta_features[val_idx, i] = base_learner.predict(X.iloc[val_idx])
                except Exception as e:
                    logger.warning(f"Base learner {i} CV fold {fold} failed: {e}")
                    meta_features[val_idx, i] = y_fold_train.mean()
        
        # Step 2: Train all base learners on full training data
        logger.debug("Training base learners on full data...")
        for i, base_learner in enumerate(self.base_learners):
            try:
                base_learner.fit(X, y_train)
                logger.debug(f"Base learner {i} trained")
            except Exception as e:
                logger.warning(f"Base learner {i} training failed: {e}")
        
        # Step 3: Train meta-learner on meta-features
        logger.debug("Training meta-learner...")
        self.meta_features_shape = meta_features.shape[1]
        
        if self.meta_learner_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        elif self.meta_learner_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_learner = LinearRegression()
        elif self.meta_learner_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.meta_learner = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=self.random_state)
        elif XGBOOST_AVAILABLE and self.meta_learner_type == 'xgb':
            self.meta_learner = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=self.random_state)
        else:
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        
        self.meta_learner.fit(meta_features, y_train.values)
        
        # Diagnostics
        train_pred = self.predict(X)
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        logger.info(f"Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logger.info("Stacking ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained ensemble.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predictions
        """
        if self.meta_learner is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(
            X.copy(),
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        feature_names = FeatureEngineer.get_feature_names(X_engineered, exclude_cols=exclude_cols)
        X_features = X_engineered[feature_names].copy()
        X_features = self._preprocess_features(X_features)
        
        # Generate meta-features from base learners
        meta_features = np.zeros((len(X), self.meta_features_shape))
        
        for i, base_learner in enumerate(self.base_learners):
            try:
                meta_features[:, i] = base_learner.predict(X)
            except Exception as e:
                logger.warning(f"Base learner {i} prediction failed: {e}")
                meta_features[:, i] = 0
        
        # Meta-learner prediction
        predictions = self.meta_learner.predict(meta_features)
        
        # Enforce non-negative
        if self.enforce_non_negative:
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def _preprocess_features(self, X_features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for training."""
        X_features = X_features.ffill(limit=10)
        X_features = X_features.bfill(limit=10)
        X_features = X_features.fillna(0)
        
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        return X_features
    
    def _get_cv_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation splits."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        return list(kf.split(np.arange(n_samples)))
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'model_name': self.model_name,
            'n_base_learners': len(self.base_learners),
            'meta_learner': self.meta_learner_type,
            'cv_splits': self.cv_splits,
            'base_learners': [bl.model_name for bl in self.base_learners],
        }


class VotingEnsembleRegressor(BaseForecastingModel):
    """
    Voting Ensemble combining predictions via weighted averaging.
    
    Simple ensemble that averages predictions from multiple models
    with optional weighting. Faster than stacking but less sophisticated.
    
    Configuration:
    - weights: List of weights for each model (default: equal)
    - aggregation: 'mean', 'median', 'weighted_mean' (default: 'mean')
    - random_state: Reproducibility seed (default: 42)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Voting Ensemble."""
        super().__init__(model_name="VotingEnsemble", config=config)
        
        self.config = config or {}
        self.weights = self.config.get('weights', None)
        self.aggregation = self.config.get('aggregation', 'mean')
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.random_state = self.config.get('random_state', 42)
        self.min_samples = self.config.get('min_samples', 10)
        
        # Feature engineering
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        # Initialize base learners
        self.base_learners = []
        self._init_base_learners()
        
        # Set default weights if not provided
        if self.weights is None:
            self.weights = np.ones(len(self.base_learners)) / len(self.base_learners)
        else:
            self.weights = np.array(self.weights) / np.sum(self.weights)
        
        self.feature_names = []
        
        logger.info(
            f"VotingEnsembleRegressor initialized: aggregation={self.aggregation}, "
            f"n_base_learners={len(self.base_learners)}"
        )
    
    def _init_base_learners(self):
        """Initialize base learners."""
        self.base_learners = [
            LinearRegressionBaseline(config={'enforce_non_negative': False}),
            RandomForestBaseline(config={'n_estimators': 100, 'enforce_non_negative': False}),
            SeasonalNaiveModel(config={'seasonal_period': 168, 'fallback_method': 'simple_naive'}),
        ]
        
        if XGBOOST_AVAILABLE:
            try:
                from src.models.baseline.xgboost_models import XGBoostBaseline
                self.base_learners.append(
                    XGBoostBaseline(config={'n_estimators': 100, 'enforce_non_negative': False})
                )
            except Exception as e:
                logger.warning(f"Could not add XGBoost to base learners: {e}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'VotingEnsembleRegressor':
        """
        Fit all base learners.
        
        Args:
            X: Feature dataframe
            y: Target values
            
        Returns:
            self for method chaining
        """
        self.validate_input(X, y)
        logger.info(f"Fitting VotingEnsembleRegressor on {len(X)} samples")
        
        if len(X) < self.min_samples:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples}")
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(X.copy(), handle_missing=True)
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(X_engineered, exclude_cols=exclude_cols)
        
        # Train all base learners
        for i, base_learner in enumerate(self.base_learners):
            try:
                logger.debug(f"Training base learner {i}...")
                base_learner.fit(X, y)
            except Exception as e:
                logger.warning(f"Base learner {i} training failed: {e}")
        
        # Diagnostics
        train_pred = self.predict(X)
        train_r2 = r2_score(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        
        logger.info(f"Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions by aggregating base learner predictions.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predictions
        """
        self.validate_input(X)
        
        predictions_list = []
        
        for i, base_learner in enumerate(self.base_learners):
            try:
                pred = base_learner.predict(X)
                predictions_list.append(pred)
            except Exception as e:
                logger.warning(f"Base learner {i} prediction failed: {e}")
                predictions_list.append(np.zeros(len(X)))
        
        predictions_array = np.column_stack(predictions_list)
        
        # Aggregate predictions
        if self.aggregation == 'mean':
            final_predictions = np.mean(predictions_array, axis=1)
        elif self.aggregation == 'median':
            final_predictions = np.median(predictions_array, axis=1)
        elif self.aggregation == 'weighted_mean':
            # Ensure weights are correctly normalized
            w = self.weights[:predictions_array.shape[1]]  # Match dimensions
            w = w / np.sum(w)  # Renormalize
            final_predictions = np.average(predictions_array, axis=1, weights=w)
        else:
            final_predictions = np.mean(predictions_array, axis=1)
        
        # Enforce non-negative
        if self.enforce_non_negative:
            final_predictions = np.maximum(final_predictions, 0)
        
        return final_predictions
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'model_name': self.model_name,
            'n_base_learners': len(self.base_learners),
            'aggregation': self.aggregation,
            'weights': self.weights.tolist(),
            'base_learners': [bl.model_name for bl in self.base_learners],
        }


class BlendingEnsembleRegressor(BaseForecastingModel):
    """
    Blending Ensemble using holdout validation set.
    
    Similar to stacking but uses a single holdout set instead of
    cross-validation for generating meta-features.
    
    Faster than stacking but may have higher variance.
    
    Configuration:
    - holdout_ratio: Fraction for holdout set (default: 0.2)
    - meta_learner_type: 'linear', 'ridge', 'rf', 'xgb'
    - random_state: Reproducibility seed
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Blending Ensemble."""
        super().__init__(model_name="BlendingEnsemble", config=config)
        
        self.config = config or {}
        self.holdout_ratio = self.config.get('holdout_ratio', 0.2)
        self.meta_learner_type = self.config.get('meta_learner_type', 'ridge')
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.random_state = self.config.get('random_state', 42)
        self.min_samples = self.config.get('min_samples', 20)
        
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        self.base_learners = []
        self._init_base_learners()
        
        self.meta_learner = None
        self.feature_names = []
        self.meta_features_shape = None
        
        logger.info(
            f"BlendingEnsembleRegressor initialized: holdout_ratio={self.holdout_ratio}, "
            f"meta_learner={self.meta_learner_type}"
        )
    
    def _init_base_learners(self):
        """Initialize base learners."""
        self.base_learners = [
            LinearRegressionBaseline(config={'enforce_non_negative': False}),
            RandomForestBaseline(config={'n_estimators': 100, 'enforce_non_negative': False}),
            SeasonalNaiveModel(config={'seasonal_period': 168, 'fallback_method': 'simple_naive'}),
        ]
        
        if XGBOOST_AVAILABLE:
            try:
                from src.models.baseline.xgboost_models import XGBoostBaseline
                self.base_learners.append(
                    XGBoostBaseline(config={'n_estimators': 100, 'enforce_non_negative': False})
                )
            except:
                pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BlendingEnsembleRegressor':
        """Fit blending ensemble."""
        self.validate_input(X, y)
        logger.info(f"Fitting BlendingEnsembleRegressor on {len(X)} samples")
        
        if len(X) < self.min_samples * 2:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples * 2}")
        
        # Split data
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.holdout_ratio, random_state=self.random_state
        )
        
        logger.info(f"Split: {len(X_train)} train, {len(X_blend)} blend")
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(X.copy(), handle_missing=True)
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(X_engineered, exclude_cols=exclude_cols)
        
        # Generate blend features
        meta_features_blend = np.zeros((len(X_blend), len(self.base_learners)))
        
        # Train base learners and generate blend features
        for i, base_learner in enumerate(self.base_learners):
            try:
                logger.debug(f"Training base learner {i}...")
                base_learner.fit(X_train, y_train)
                meta_features_blend[:, i] = base_learner.predict(X_blend)
            except Exception as e:
                logger.warning(f"Base learner {i} failed: {e}")
                meta_features_blend[:, i] = y_blend.mean()
        
        # Retrain all on full dataset
        for base_learner in self.base_learners:
            try:
                base_learner.fit(X, y)
            except Exception as e:
                logger.warning(f"Retrain failed: {e}")
        
        # Train meta-learner
        self._init_meta_learner()
        self.meta_features_shape = meta_features_blend.shape[1]
        self.meta_learner.fit(meta_features_blend, y_blend.values)
        
        logger.info("Blending ensemble training completed")
        return self
    
    def _init_meta_learner(self):
        """Initialize meta-learner."""
        if self.meta_learner_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        elif self.meta_learner_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_learner = LinearRegression()
        elif self.meta_learner_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.meta_learner = RandomForestRegressor(n_estimators=50, max_depth=6)
        elif XGBOOST_AVAILABLE and self.meta_learner_type == 'xgb':
            self.meta_learner = xgb.XGBRegressor(n_estimators=50, max_depth=4)
        else:
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.meta_learner is None:
            raise RuntimeError("Model must be fitted")
        
        meta_features = np.zeros((len(X), self.meta_features_shape))
        
        for i, base_learner in enumerate(self.base_learners):
            try:
                meta_features[:, i] = base_learner.predict(X)
            except Exception as e:
                logger.warning(f"Base learner {i} failed: {e}")
                meta_features[:, i] = 0
        
        predictions = self.meta_learner.predict(meta_features)
        
        if self.enforce_non_negative:
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'model_name': self.model_name,
            'n_base_learners': len(self.base_learners),
            'holdout_ratio': self.holdout_ratio,
            'meta_learner': self.meta_learner_type,
        }


class OptimalWeightingEnsemble(BaseForecastingModel):
    """
    Ensemble with optimal weight learning.
    
    Uses constrained optimization to find optimal weights
    for combining base learner predictions.
    
    Minimizes MSE subject to: sum(weights) = 1, weights >= 0
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Optimal Weighting Ensemble."""
        super().__init__(model_name="OptimalWeightingEnsemble", config=config)
        
        self.config = config or {}
        self.enforce_non_negative = self.config.get('enforce_non_negative', True)
        self.random_state = self.config.get('random_state', 42)
        self.min_samples = self.config.get('min_samples', 20)
        
        fe_config = {
            'lag_periods': self.config.get('lag_periods', [1, 7, 168]),
            'rolling_windows': self.config.get('rolling_windows', [6, 24, 168]),
            'seasonal_periods': self.config.get('seasonal_periods', [24, 168]),
        }
        self.feature_engineer = FeatureEngineer(config=fe_config)
        
        self.base_learners = []
        self._init_base_learners()
        
        self.optimal_weights = None
        self.feature_names = []
        
        logger.info("OptimalWeightingEnsemble initialized")
    
    def _init_base_learners(self):
        """Initialize base learners."""
        self.base_learners = [
            LinearRegressionBaseline(config={'enforce_non_negative': False}),
            RandomForestBaseline(config={'n_estimators': 100, 'enforce_non_negative': False}),
            SeasonalNaiveModel(config={'seasonal_period': 168, 'fallback_method': 'simple_naive'}),
        ]
        
        if XGBOOST_AVAILABLE:
            try:
                from src.models.baseline.xgboost_models import XGBoostBaseline
                self.base_learners.append(
                    XGBoostBaseline(config={'n_estimators': 100, 'enforce_non_negative': False})
                )
            except:
                pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'OptimalWeightingEnsemble':
        """Fit ensemble with optimal weighting."""
        self.validate_input(X, y)
        logger.info(f"Fitting OptimalWeightingEnsemble on {len(X)} samples")
        
        if len(X) < self.min_samples:
            raise ValueError(f"Insufficient samples")
        
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_all_features(X.copy(), handle_missing=True)
        exclude_cols = ['store_id', 'product_id', 'city_id', 'dt', 'sale_amount']
        self.feature_names = FeatureEngineer.get_feature_names(X_engineered, exclude_cols=exclude_cols)
        
        # Train base learners
        base_predictions = np.zeros((len(X), len(self.base_learners)))
        
        for i, base_learner in enumerate(self.base_learners):
            try:
                base_learner.fit(X, y)
                base_predictions[:, i] = base_learner.predict(X)
            except Exception as e:
                logger.warning(f"Base learner {i} failed: {e}")
                base_predictions[:, i] = y.mean()
        
        # Optimize weights
        from scipy.optimize import minimize
        
        def mse_loss(weights, X, y):
            """MSE loss for weight optimization."""
            predictions = np.dot(X, weights)
            return mean_squared_error(y, predictions)
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.base_learners))]
        x0 = np.ones(len(self.base_learners)) / len(self.base_learners)
        
        result = minimize(
            mse_loss,
            x0,
            args=(base_predictions, y.values),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.optimal_weights = result.x
        logger.info(f"Optimal weights: {self.optimal_weights}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with optimal weights."""
        if self.optimal_weights is None:
            raise RuntimeError("Model must be fitted")
        
        base_predictions = np.zeros((len(X), len(self.base_learners)))
        
        for i, base_learner in enumerate(self.base_learners):
            try:
                base_predictions[:, i] = base_learner.predict(X)
            except Exception as e:
                logger.warning(f"Base learner {i} failed: {e}")
                base_predictions[:, i] = 0
        
        predictions = np.dot(base_predictions, self.optimal_weights)
        
        if self.enforce_non_negative:
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'model_name': self.model_name,
            'n_base_learners': len(self.base_learners),
            'optimal_weights': self.optimal_weights.tolist() if self.optimal_weights is not None else None,
        }
