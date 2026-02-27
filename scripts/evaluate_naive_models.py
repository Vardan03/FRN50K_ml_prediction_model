#!/usr/bin/env python3
# scripts/evaluate_naive_models.py
"""
Evaluation script for naive forecasting models.

This script:
1. Loads training and evaluation data
2. Trains three naive models
3. Generates predictions on eval set
4. Computes performance metrics
5. Compares results and provides insights
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.evaluate.metrics import RegressionMetrics
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


class NaiveModelEvaluator:
    """Comprehensive evaluator for naive forecasting models."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            data_dir: Directory containing train and eval data
            output_dir: Directory to save results (default: models/trained/)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or str(project_root / 'models' / 'trained')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DataLoader(data_dir)
        self.metrics_calculator = RegressionMetrics()
        self.results = {}
        
        logger.info(f"Initialized NaiveModelEvaluator with output_dir={self.output_dir}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load training and evaluation data.
        
        Returns:
            Tuple of (X_train, y_train, X_eval, y_eval)
        """
        logger.info("Loading data...")
        
        # Load train data
        train_data = self.data_loader.load_train_data()
        X_train = train_data.drop('sale_amount', axis=1)
        y_train = train_data['sale_amount']
        
        logger.info(f"Train data: {X_train.shape}")
        logger.debug(f"Train columns: {X_train.columns.tolist()}")
        
        # Load eval data
        eval_data = self.data_loader.load_eval_data()
        X_eval = eval_data.drop('sale_amount', axis=1)
        y_eval = eval_data['sale_amount']
        
        logger.info(f"Eval data: {X_eval.shape}")
        
        return X_train, y_train, X_eval, y_eval
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train all three naive models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training naive models...")
        
        models = {}
        
        # 1. Simple Naive
        logger.info("Training SimpleNaiveModel...")
        try:
            model_simple = SimpleNaiveModel(config={
                'fill_method': 'forward_fill',
                'default_value': 0.0
            })
            model_simple.fit(X_train, y_train)
            models['simple'] = model_simple
            logger.info(f"✓ SimpleNaiveModel trained: {model_simple.get_model_summary()}")
        except Exception as e:
            logger.error(f"✗ Failed to train SimpleNaiveModel: {e}")
        
        # 2. Seasonal Naive
        logger.info("Training SeasonalNaiveModel...")
        try:
            model_seasonal = SeasonalNaiveModel(config={
                'seasonal_period': 168,  # 1 week
                'fallback_method': 'simple_naive',
                'default_value': 0.0
            })
            model_seasonal.fit(X_train, y_train)
            models['seasonal'] = model_seasonal
            logger.info(f"✓ SeasonalNaiveModel trained: {model_seasonal.get_model_summary()}")
        except Exception as e:
            logger.error(f"✗ Failed to train SeasonalNaiveModel: {e}")
        
        # 3. Weighted Moving Average
        logger.info("Training WeightedMovingAverageModel...")
        try:
            model_wma = WeightedMovingAverageModel(config={
                'windows': [6, 24, 168],
                'weights': 'exponential',
                'alpha': 0.7,
                'default_value': 0.0
            })
            model_wma.fit(X_train, y_train)
            models['wma'] = model_wma
            logger.info(f"✓ WeightedMovingAverageModel trained: {model_wma.get_model_summary()}")
        except Exception as e:
            logger.error(f"✗ Failed to train WeightedMovingAverageModel: {e}")
        
        logger.info(f"Successfully trained {len(models)} models")
        return models
    
    def evaluate_models(self, models: Dict, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict:
        """
        Evaluate all trained models on evaluation set.
        
        Args:
            models: Dictionary of trained models
            X_eval: Evaluation features
            y_eval: Evaluation target
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating models on eval set...")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                # Generate predictions
                y_pred = model.predict(X_eval)
                
                # Compute metrics
                metrics = self.metrics_calculator.compute_all_metrics(y_eval, y_pred)
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'status': 'success'
                }
                
                logger.info(f"✓ {model_name} evaluation complete")
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  MAPE: {metrics['mape']:.4f}%")
                logger.info(f"  R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"✗ Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return evaluation_results
    
    def compare_results(self, evaluation_results: Dict) -> None:
        """
        Compare and display model performance.
        
        Args:
            evaluation_results: Dictionary of evaluation results
        """
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("=" * 80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, result in evaluation_results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape'],
                    'R²': metrics['r2'],
                    'Median APE': metrics['median_ape']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display results
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Identify best models
        if len(comparison_df) > 0:
            best_rmse_idx = comparison_df['RMSE'].idxmin()
            best_mae_idx = comparison_df['MAE'].idxmin()
            best_r2_idx = comparison_df['R²'].idxmax()
            
            logger.info("\n" + "=" * 80)
            logger.info("BEST PERFORMERS:")
            logger.info(f"  Lowest RMSE: {comparison_df.loc[best_rmse_idx, 'Model']} "
                       f"({comparison_df.loc[best_rmse_idx, 'RMSE']:.4f})")
            logger.info(f"  Lowest MAE: {comparison_df.loc[best_mae_idx, 'Model']} "
                       f"({comparison_df.loc[best_mae_idx, 'MAE']:.4f})")
            logger.info(f"  Highest R²: {comparison_df.loc[best_r2_idx, 'Model']} "
                       f"({comparison_df.loc[best_r2_idx, 'R²']:.4f})")
            logger.info("=" * 80)
        
        self.comparison_df = comparison_df
    
    def save_results(self, models: Dict, evaluation_results: Dict) -> None:
        """
        Save models and results to disk.
        
        Args:
            models: Trained models
            evaluation_results: Evaluation results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for model_name, model in models.items():
            try:
                filepath = str(Path(self.output_dir) / f"{model_name}_model_{timestamp}.joblib")
                model.save_model(filepath)
                logger.info(f"Saved {model_name} model to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {model_name} model: {e}")
        
        # Save results
        try:
            results_filepath = str(Path(self.output_dir) / f"naive_models_results_{timestamp}.json")
            
            # Prepare results for JSON serialization
            json_results = {}
            for model_name, result in evaluation_results.items():
                if result['status'] == 'success':
                    json_results[model_name] = {
                        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in result['metrics'].items()}
                    }
                else:
                    json_results[model_name] = {'status': 'failed', 'error': result.get('error')}
            
            with open(results_filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved results to {results_filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Save comparison dataframe
        try:
            csv_filepath = str(Path(self.output_dir) / f"naive_models_comparison_{timestamp}.csv")
            self.comparison_df.to_csv(csv_filepath, index=False)
            logger.info(f"Saved comparison to {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")
    
    def run_evaluation(self) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary of all results
        """
        try:
            # Load data
            X_train, y_train, X_eval, y_eval = self.load_data()
            
            # Train models
            models = self.train_models(X_train, y_train)
            
            if not models:
                logger.error("No models were successfully trained")
                return {}
            
            # Evaluate models
            evaluation_results = self.evaluate_models(models, X_eval, y_eval)
            
            # Compare results
            self.compare_results(evaluation_results)
            
            # Save results
            self.save_results(models, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
            return {}


def main():
    """Main execution function."""
    # Determine data directory
    data_dir = str(project_root / 'dataset' / 'raw')
    
    logger.info("Starting naive models evaluation...")
    logger.info(f"Data directory: {data_dir}")
    
    # Create evaluator
    evaluator = NaiveModelEvaluator(data_dir)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        logger.info("✓ Evaluation completed successfully!")
    else:
        logger.error("✗ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
