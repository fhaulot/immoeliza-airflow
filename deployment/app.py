"""
FastAPI application for Belgian Real Estate Price Prediction
Updated to work with the harmonized Immovlan pipeline
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import uvicorn
import joblib
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from model.Preprocessing import ModelPreprocessing

# Create FastAPI app
app = FastAPI(
    title="Immovlan Real Estate Price Prediction API",
    description="Machine learning-based price predictions for Belgian real estate properties using harmonized Immovlan data",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing
model = None
preprocessor = None
feature_columns = None
model_metadata = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessing components"""
    global model, preprocessor, feature_columns, model_metadata
    
    try:
        # Paths
        model_dir = Path(__file__).parent.parent / "model" / "trained_models"
        processed_data_dir = Path(__file__).parent.parent / "model" / "processed_data"
        
        # Load model
        model_path = model_dir / "best_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Load preprocessor
        preprocessor = ModelPreprocessing()
        preprocessor.load_preprocessor(processed_data_dir)
        
        # Load feature columns
        feature_cols_path = processed_data_dir / "feature_columns.csv"
        if feature_cols_path.exists():
            feature_columns = pd.read_csv(feature_cols_path).iloc[:, 0].tolist()
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        else:
            logger.error(f"Feature columns file not found: {feature_cols_path}")
            return False
        
        # Load model metadata
        metadata_path = model_dir / "model_metadata.csv"
        if metadata_path.exists():
            model_metadata = pd.read_csv(metadata_path).iloc[0].to_dict()
            logger.info(f"Loaded model metadata: {model_metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    success = load_model_and_preprocessor()
    if success:
        logger.info("Application started successfully")
    else:
        logger.error("Failed to load model and preprocessor")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    # Core features based on our Immovlan data structure
    habitableSurface: float = Field(..., description="Habitable surface area (m²)", gt=0, le=1000)
    bedroomCount: int = Field(..., description="Number of bedrooms", ge=1, le=20)
    postCode: int = Field(..., description="Belgian postal code", ge=1000, le=9999)
    type: str = Field(..., description="Property type (APARTMENT or HOUSE)")
    subtype: str = Field(..., description="Property subtype")
    province: str = Field(..., description="Belgian province")
    region: str = Field(..., description="Belgian region (Brussels, Flanders, Wallonia)")
    
    # Optional features
    hasGarden: Optional[int] = Field(0, description="Has garden (0/1)")
    gardenSurface: Optional[float] = Field(0.0, description="Garden surface area (m²)")
    hasTerrace: Optional[int] = Field(0, description="Has terrace (0/1)")
    hasParking: Optional[int] = Field(0, description="Has parking (0/1)")
    buildingCondition: Optional[str] = Field("UNKNOWN", description="Building condition")
    epcScore: Optional[str] = Field("UNKNOWN", description="Energy performance certificate (A-G)")

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted price in EUR")
    currency: str = Field("EUR", description="Currency")
    confidence_score: Optional[float] = Field(None, description="Model confidence (R² score)")
    model_name: Optional[str] = Field(None, description="Name of the model used")
    status: str = Field("success", description="Status")
    timestamp: str = Field(..., description="Prediction timestamp")
    input_summary: Dict[str, Any] = Field(..., description="Input parameters summary")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    feature_count: Optional[int]
    model_info: Optional[Dict[str, Any]]
    timestamp: str

def validate_and_prepare_input(request: PredictionRequest) -> pd.DataFrame:
    """
    Validate and prepare input data for prediction
    """
    # Convert request to dictionary
    input_data = request.dict()
    
    # Validate province and region consistency
    province_region_map = {
        "Brussels": "Brussels",
        "Vlaams-Brabant": "Flanders", "Antwerp": "Flanders", "Limburg": "Flanders",
        "East-Flanders": "Flanders", "West-Flanders": "Flanders",
        "Brabant Wallon": "Wallonia", "Hainaut": "Wallonia", "Liège": "Wallonia",
        "Namur": "Wallonia", "Luxembourg": "Wallonia"
    }
    
    expected_region = province_region_map.get(input_data["province"])
    if expected_region and input_data["region"] != expected_region:
        raise HTTPException(
            status_code=400, 
            detail=f"Province {input_data['province']} should be in region {expected_region}, not {input_data['region']}"
        )
    
    # Validate postal code ranges
    postal_code = input_data["postCode"]
    if not (1000 <= postal_code <= 9999):
        raise HTTPException(status_code=400, detail="Postal code must be between 1000 and 9999")
    
    # Create DataFrame with single row
    df = pd.DataFrame([input_data])
    
    return df

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data for prediction using our trained preprocessor
    """
    # Apply ordinal encodings
    df = preprocessor.ordinal_building_condition(df)
    df = preprocessor.ordinal_epc_score(df)
    
    # Apply categorical encoding (one-hot)
    df = preprocessor.categorical_encode(df)
    
    # Ensure all required feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    # Select only the required feature columns in the correct order
    df = df[feature_columns]
    
    # Scale features using the trained scaler
    if preprocessor.scaler is not None:
        df_scaled = preprocessor.scaler.transform(df)
        df = pd.DataFrame(df_scaled, columns=feature_columns)
    
    return df

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = f"""
    <html>
        <head>
            <title>Immovlan Price Prediction API</title>
        </head>
        <body>
            <h1>Immovlan Real Estate Price Prediction API</h1>
            <p>This API provides machine learning-based price predictions for Belgian real estate properties.</p>
            <p><strong>Version:</strong> 2.0.0</p>
            <p><strong>Model Status:</strong> {'✅ Loaded' if model is not None else '❌ Not Loaded'}</p>
            <p><strong>Endpoints:</strong></p>
            <ul>
                <li><a href="/docs">📖 API Documentation</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
                <li><a href="/predict">🔮 Prediction (POST)</a></li>
                <li><a href="/model-info">ℹ️ Model Information</a></li>
            </ul>
        </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        feature_count=len(feature_columns) if feature_columns else None,
        model_info=model_metadata,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": model_metadata,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "feature_columns": feature_columns[:10] if feature_columns else [],  # Show first 10
        "preprocessing_info": {
            "scaler_loaded": preprocessor.scaler is not None if preprocessor else False,
            "ordinal_encodings": ["buildingCondition", "epcScore"]
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict real estate price based on property features
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    try:
        # Validate and prepare input
        input_df = validate_and_prepare_input(request)
        
        # Preprocess for prediction
        processed_df = preprocess_for_prediction(input_df)
        
        # Make prediction
        prediction = model.predict(processed_df)[0]
        
        # Ensure prediction is positive
        prediction = max(prediction, 50000)  # Minimum reasonable price
        
        response = PredictionResponse(
            predicted_price=float(prediction),
            currency="EUR",
            confidence_score=model_metadata.get("test_r2") if model_metadata else None,
            model_name=model_metadata.get("model_name") if model_metadata else "Random Forest",
            status="success",
            timestamp=datetime.now().isoformat(),
            input_summary=request.dict()
        )
        
        logger.info(f"Prediction made: {prediction:.0f}€ for {request.type} in {request.province}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/example-request")
async def get_example_request():
    """Get an example request for testing"""
    return {
        "habitableSurface": 120.0,
        "bedroomCount": 3,
        "postCode": 1000,
        "type": "HOUSE",
        "subtype": "HOUSE",
        "province": "Brussels",
        "region": "Brussels",
        "hasGarden": 1,
        "gardenSurface": 50.0,
        "hasTerrace": 1,
        "hasParking": 1,
        "buildingCondition": "GOOD",
        "epcScore": "C"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )