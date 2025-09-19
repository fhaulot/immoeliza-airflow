import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPreprocessing: 

    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.feature_columns = None

    def read_csv(self, path):
        """
        Reading the scraped Immovlan data csv file for model training
        """
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    
    def clean_for_model(self, df):
        """
        Aggressive cleaning for ML model training - stricter than analysis
        """
        logger.info("Starting model-specific data cleaning...")
        initial_rows = len(df)
        
        # Remove rows with missing critical data for ML
        required_columns = ['price', 'postCode', 'habitableSurface', 'bedroomCount']
        df = df.dropna(subset=required_columns)
        logger.info(f"Removed {initial_rows - len(df)} rows with missing critical data")
        
        # More aggressive price filtering for model
        df = df[(df['price'] >= 50000) & (df['price'] <= 2000000)]
        logger.info(f"Applied strict price filtering, remaining rows: {len(df)}")
        
        # More aggressive surface filtering
        df = df[(df['habitableSurface'] >= 20) & (df['habitableSurface'] <= 500)]
        logger.info(f"Applied strict surface filtering, remaining rows: {len(df)}")
        
        # Filter bedroom counts
        df = df[(df['bedroomCount'] >= 1) & (df['bedroomCount'] <= 10)]
        logger.info(f"Applied bedroom filtering, remaining rows: {len(df)}")
        
        # Remove extreme price per square meter outliers
        if 'price_square_meter' in df.columns:
            q1 = df['price_square_meter'].quantile(0.01)
            q99 = df['price_square_meter'].quantile(0.99)
            df = df[(df['price_square_meter'] >= q1) & (df['price_square_meter'] <= q99)]
            logger.info(f"Removed price/sqm outliers, remaining rows: {len(df)}")
        
        return df
    
    def prepare_features_for_model(self, df):
        """
        Prepare features specifically for ML model training
        """
        logger.info("Preparing features for ML model...")
        
        # Select features for ML model
        numeric_features = [
            'habitableSurface', 'bedroomCount', 'postCode',
            'hasGarden', 'gardenSurface', 'hasTerrace', 'hasParking'
        ]
        
        categorical_features = [
            'type', 'subtype', 'province', 'region', 'buildingCondition', 'epcScore'
        ]
        
        # Filter to existing columns
        numeric_features = [col for col in numeric_features if col in df.columns]
        categorical_features = [col for col in categorical_features if col in df.columns]
        
        # Handle missing values in categorical features
        for col in categorical_features:
            df[col] = df[col].fillna('UNKNOWN')
        
        # Handle missing values in numeric features
        for col in numeric_features:
            if col in ['gardenSurface']:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Create final feature set
        features_df = df[numeric_features + categorical_features].copy()
        target = df['price'].copy()
        
        logger.info(f"Selected {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        logger.info(f"Feature columns: {numeric_features + categorical_features}")
        
        return features_df, target
    
    def ordinal_building_condition(self, df):
        """
        Perform ordinal encoding for building condition (consistent with analysis)
        """
        condition_order = {
            "AS_NEW": 1, 
            "GOOD": 2, 
            "JUST_RENOVATED": 3,
            "TO_RENOVATE": 4, 
            "TO_BE_DONE_UP": 5, 
            "TO_RESTORE": 6,
            "UNKNOWN": 3
        }
        df['buildingCondition'] = df['buildingCondition'].map(condition_order)
        df['buildingCondition'] = df['buildingCondition'].fillna(3)
        logger.info("Applied ordinal encoding to building condition")
        return df
    
    def ordinal_epc_score(self, df):
        """
        Perform ordinal encoding for EPC score (consistent with analysis)
        """
        epc_order = {
            "A": 1, "B": 2, "C": 3, "D": 4, 
            "E": 5, "F": 6, "G": 7, "UNKNOWN": 4
        }
        df['epcScore'] = df['epcScore'].map(epc_order)
        df['epcScore'] = df['epcScore'].fillna(4)
        logger.info("Applied ordinal encoding to EPC score")
        return df
    
    def categorical_encode(self, df):
        """
        Perform categorical encoding for ML model training with proper handling
        """
        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_columns:
            logger.info(f"Encoding categorical columns: {categorical_columns}")
            
            # Use get_dummies for simplicity and consistency
            df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
            
            # Convert boolean columns to int
            bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
            df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
            
            logger.info(f"Final encoded shape: {df_encoded.shape}")
            return df_encoded
        else:
            logger.info("No categorical columns found for encoding")
            return df
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features for ML model training
        """
        if fit:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("Fitted scaler on training data")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_train_scaled = self.scaler.transform(X_train)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def train_test_split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, save_dir):
        """
        Save the preprocessing components for later use
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / 'scaler.pkl')
            logger.info(f"Saved scaler to {save_dir / 'scaler.pkl'}")
        
        if self.feature_columns is not None:
            pd.Series(self.feature_columns).to_csv(save_dir / 'feature_columns.csv', index=False)
            logger.info(f"Saved feature columns to {save_dir / 'feature_columns.csv'}")
    
    def load_preprocessor(self, save_dir):
        """
        Load preprocessing components
        """
        save_dir = Path(save_dir)
        
        scaler_path = save_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        feature_cols_path = save_dir / 'feature_columns.csv'
        if feature_cols_path.exists():
            self.feature_columns = pd.read_csv(feature_cols_path).iloc[:, 0].tolist()
            logger.info(f"Loaded feature columns from {feature_cols_path}")
    
    def full_preprocessing_pipeline(self, input_path, output_dir):
        """
        Run the complete preprocessing pipeline for model training
        """
        logger.info("=== Starting Model Preprocessing Pipeline ===")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load data
        df = self.read_csv(input_path)
        
        # Clean data for model
        df = self.clean_for_model(df)
        
        # Apply ordinal encoding
        df = self.ordinal_building_condition(df)
        df = self.ordinal_epc_score(df)
        
        # Prepare features and target
        X, y = self.prepare_features_for_model(df)
        
        # Apply categorical encoding
        X = self.categorical_encode(X)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # Save processed data
        X_train_scaled.to_csv(output_dir / 'X_train.csv', index=False)
        X_test_scaled.to_csv(output_dir / 'X_test.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
        y_test.to_csv(output_dir / 'y_test.csv', index=False)
        
        # Save preprocessor components
        self.save_preprocessor(output_dir)
        
        logger.info(f"Model-ready data saved to: {output_dir}")
        logger.info(f"Final feature count: {len(self.feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
