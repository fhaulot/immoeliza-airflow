"""
Test script for the complete ML pipeline
Tests: Scraping -> Analysis -> Model Training -> Deployment
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
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analyse.Preprocessing import AnalysisPreprocessing
from model.Preprocessing import ModelPreprocessing
from model.pipeline import ModelPipeline

def test_full_pipeline():
    """Test the complete pipeline"""
    
    logger.info("="*80)
    logger.info("TESTING COMPLETE ML PIPELINE")
    logger.info("="*80)
    
    # Define paths - check multiple possible locations
    scraped_data_paths = [
        project_root / "data" / "immovlan_scraped_data.csv",
        project_root / "immovlan_scraped_data_test.csv",
        project_root / "immovlan_scraped_data.csv",
    ]
    
    scraped_data = None
    for path in scraped_data_paths:
        if path.exists():
            scraped_data = path
            logger.info(f"Found scraped data at: {scraped_data}")
            break
    
    if scraped_data is None:
        logger.error(f"No scraped data found! Checked: {[str(p) for p in scraped_data_paths]}")
        return
        return False
    
    analyse_output = project_root / "analyse" / "processed_for_analysis.csv"
    model_data_dir = project_root / "model" / "processed_data"
    model_output_dir = project_root / "model" / "trained_models"
    
    # Step 1: Analysis Preprocessing
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Analysis Preprocessing")
    logger.info("="*80)
    try:
        analysis_preprocessor = AnalysisPreprocessing()
        df_analysis = analysis_preprocessor.full_preprocessing_pipeline(
            input_path=str(scraped_data),
            output_path=str(analyse_output)
        )
        logger.info(f"✓ Analysis preprocessing complete. Output shape: {df_analysis.shape}")
    except Exception as e:
        logger.error(f"✗ Analysis preprocessing failed: {e}")
        return False
    
    # Step 2: Model Preprocessing
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Model Preprocessing")
    logger.info("="*80)
    try:
        model_preprocessor = ModelPreprocessing()
        X_train, X_test, y_train, y_test = model_preprocessor.full_preprocessing_pipeline(
            input_path=str(scraped_data),
            output_dir=str(model_data_dir)
        )
        logger.info(f"✓ Model preprocessing complete.")
        logger.info(f"  Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        if len(X_train) < 20:
            logger.warning(f"⚠ WARNING: Very small training set ({len(X_train)} samples). Results may not be reliable.")
            logger.warning("  Recommendation: Run full scraper to get more data for better model training.")
    except Exception as e:
        logger.error(f"✗ Model preprocessing failed: {e}")
        return False
    
    # Step 3: Model Training
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Model Training")
    logger.info("="*80)
    try:
        pipeline = ModelPipeline()
        pipeline.create_pipelines()
        
        results = pipeline.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        logger.info(f"✓ Model training complete. Tested {len(results)} models.")
        logger.info(f"  Best model: {pipeline.best_model_name}")
        if pipeline.best_score != -np.inf:
            logger.info(f"  Best R² score: {pipeline.best_score:.4f}")
        else:
            logger.warning(f"  Best R² score: {pipeline.best_score} (dataset too small for reliable metrics)")
        
        # Save models (this also saves results)
        pipeline.save_models(str(model_output_dir), results)
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Check Deployment Files
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Deployment Check")
    logger.info("="*80)
    
    best_model_path = model_output_dir / "best_model.pkl"
    scaler_path = model_data_dir / "scaler.pkl"
    encoder_path = model_data_dir / "encoder.pkl"
    
    if best_model_path.exists() and scaler_path.exists():
        logger.info(f"✓ Model files ready for deployment:")
        logger.info(f"  Model: {best_model_path}")
        logger.info(f"  Scaler: {scaler_path}")
        if encoder_path.exists():
            logger.info(f"  Encoder: {encoder_path}")
    else:
        logger.warning("⚠ Some model files missing for deployment")
        if not best_model_path.exists():
            logger.warning(f"  Missing: {best_model_path}")
        if not scaler_path.exists():
            logger.warning(f"  Missing: {scaler_path}")
        return False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("="*80)
    logger.info("✓ All steps completed successfully!")
    logger.info(f"✓ Analysis data: {analyse_output}")
    logger.info(f"✓ Model data: {model_data_dir}")
    logger.info(f"✓ Trained models: {model_output_dir}")
    logger.info(f"✓ Best model: {pipeline.best_model_name} (R²={pipeline.best_score:.4f})")
    logger.info("\nNext steps:")
    logger.info("  1. Review model performance in model/trained_models/")
    logger.info("  2. Test the API with: uv run deployment/app.py")
    logger.info("  3. Access API docs at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
