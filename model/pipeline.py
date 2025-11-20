import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPipeline:
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
    def load_preprocessed_data(self, data_dir):
        """
        Load preprocessed data from the model preprocessing step
        """
        data_dir = Path(data_dir)
        
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv').iloc[:, 0]  # Convert to Series
        y_test = pd.read_csv(data_dir / 'y_test.csv').iloc[:, 0]    # Convert to Series
        
        logger.info(f"Loaded preprocessed data:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  y_train shape: {y_train.shape}")
        logger.info(f"  y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipelines(self):
        """
        Create different regression model pipelines
        """
        self.models = {
            "Linear Regression": Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('regressor', LinearRegression())
            ]),
            
            "Ridge Regression": Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('regressor', Ridge(alpha=1.0))
            ]),
            
            "Random Forest": Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('regressor', RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10,
                    min_samples_split=5
                ))
            ]),
            
            "Gradient Boosting": Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1
                ))
            ]),
            
            "Polynomial Regression": Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
                ('regressor', Ridge(alpha=1.0))
            ])
        }
        
        logger.info(f"Created {len(self.models)} model pipelines")
        return self.models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        """
        logger.info("Training and evaluating models...")
        
        results = {}
        
        for model_name, pipeline in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train the model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                # Cross-validation score (adapt cv to dataset size)
                n_samples = len(X_train)
                cv_folds = min(5, n_samples)  # Use fewer folds for small datasets
                
                if cv_folds >= 2:
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = test_r2  # Fallback to test score
                    cv_std = 0.0
                
                results[model_name] = {
                    'pipeline': pipeline,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred
                }
                
                logger.info(f"  {model_name} - Test R²: {test_r2:.4f}, Test MAE: {test_mae:.0f}")
                
                # Track best model (handle NaN values)
                if not np.isnan(test_r2) and test_r2 > self.best_score:
                    self.best_score = test_r2
                    self.best_model = pipeline
                    self.best_model_name = model_name
                elif self.best_model is None:  # No valid model yet, use current as fallback
                    self.best_score = test_r2 if not np.isnan(test_r2) else -np.inf
                    self.best_model = pipeline
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        logger.info(f"Best model: {self.best_model_name} with R² = {self.best_score:.4f}")
        return results
    
    def print_results_summary(self, results):
        """
        Print a summary of all model results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Train R²': f"{result['train_r2']:.4f}",
                'Test R²': f"{result['test_r2']:.4f}",
                'Train MAE': f"{result['train_mae']:.0f}€",
                'Test MAE': f"{result['test_mae']:.0f}€",
                'CV R² Mean': f"{result['cv_r2_mean']:.4f}",
                'CV R² Std': f"{result['cv_r2_std']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\nBEST MODEL: {self.best_model_name}")
        print(f"Best Test R²: {self.best_score:.4f}")
        
    def save_models(self, save_dir, results):
        """
        Save the trained models and results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            model_path = save_dir / 'best_model.pkl'
            joblib.dump(self.best_model, model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {model_path}")
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'test_r2': self.best_score,
                'test_mae': results[self.best_model_name]['test_mae']
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_csv(save_dir / 'model_metadata.csv', index=False)
            logger.info(f"Saved model metadata to {save_dir / 'model_metadata.csv'}")
        
        # Save all models
        all_models_dir = save_dir / 'all_models'
        all_models_dir.mkdir(exist_ok=True)
        
        for model_name, result in results.items():
            model_filename = model_name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(result['pipeline'], all_models_dir / model_filename)
        
        logger.info(f"Saved all models to {all_models_dir}")
        
        # Save detailed results
        detailed_results = {}
        for model_name, result in results.items():
            detailed_results[model_name] = {
                k: v for k, v in result.items() 
                if k not in ['pipeline', 'y_train_pred', 'y_test_pred']
            }
        
        results_df = pd.DataFrame(detailed_results).T
        results_df.to_csv(save_dir / 'detailed_results.csv')
        logger.info(f"Saved detailed results to {save_dir / 'detailed_results.csv'}")
    
    def full_training_pipeline(self, data_dir, output_dir):
        """
        Run the complete model training pipeline
        """
        logger.info("=== Starting Model Training Pipeline ===")
        
        # Load preprocessed data
        X_train, X_test, y_train, y_test = self.load_preprocessed_data(data_dir)
        
        # Create model pipelines
        self.create_pipelines()
        
        # Train and evaluate models
        results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Print results
        self.print_results_summary(results)
        
        # Save models and results
        self.save_models(output_dir, results)
        
        logger.info(f"Training pipeline completed. Results saved to {output_dir}")
        
        return self.best_model, results

def main():
    """
    Main function to run the model training pipeline
    """
    # Paths
    data_dir = '/home/floriane/GitHub/immoeliza-airflow/model/processed_data'
    output_dir = '/home/floriane/GitHub/immoeliza-airflow/model/trained_models'
    
    # Initialize pipeline
    pipeline = ModelPipeline()
    
    try:
        # Run training pipeline
        best_model, results = pipeline.full_training_pipeline(data_dir, output_dir)
        
        print(f"\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"Best model: {pipeline.best_model_name}")
        print(f"Best R² score: {pipeline.best_score:.4f}")
        print(f"Models saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()