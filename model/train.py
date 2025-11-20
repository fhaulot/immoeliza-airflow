"""
Production Training Script
Orchestrates the full training pipeline: Analysis -> Preprocessing -> Training -> Saving
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImmoEliza-Training")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyse.Preprocessing import AnalysisPreprocessing
from model.Preprocessing import ModelPreprocessing
from model.pipeline import ModelPipeline

def run_training_pipeline():
    """Run the complete production training pipeline"""
    
    logger.info("STARTING PRODUCTION TRAINING PIPELINE")
    
    # Define paths
    # In Docker, we expect data in /app/data
    scraped_data_path = project_root / "data" / "immovlan_scraped_data.csv"
    
    if not scraped_data_path.exists():
        logger.error(f"Scraped data not found at {scraped_data_path}")
        sys.exit(1)
        
    analyse_output = project_root / "analyse" / "processed_for_analysis.csv"
    model_data_dir = project_root / "model" / "processed_data"
    model_output_dir = project_root / "model" / "trained_models"
    
    # Ensure directories exist
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analysis Preprocessing
    logger.info("STEP 1: Analysis Preprocessing")
    try:
        analysis_preprocessor = AnalysisPreprocessing()
        df_analysis = analysis_preprocessor.full_preprocessing_pipeline(
            input_path=str(scraped_data_path),
            output_path=str(analyse_output)
        )
        logger.info(f"Analysis preprocessing complete. Output shape: {df_analysis.shape}")
    except Exception as e:
        logger.error(f"Analysis preprocessing failed: {e}")
        sys.exit(1)
    
    # Step 2: Model Preprocessing
    logger.info("STEP 2: Model Preprocessing")
    try:
        model_preprocessor = ModelPreprocessing()
        X_train, X_test, y_train, y_test = model_preprocessor.full_preprocessing_pipeline(
            input_path=str(scraped_data_path),
            output_dir=str(model_data_dir)
        )
        logger.info(f"Model preprocessing complete.")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as e:
        logger.error(f"Model preprocessing failed: {e}")
        sys.exit(1)
    
    # Step 3: Model Training
    logger.info("STEP 3: Model Training")
    try:
        pipeline = ModelPipeline()
        pipeline.create_pipelines()
        
        results = pipeline.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        logger.info(f"Model training complete. Best model: {pipeline.best_model_name}")
        
        # Save models
        pipeline.save_models(str(model_output_dir), results)
        logger.info(f"Models saved to {model_output_dir}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)
    
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_training_pipeline()
