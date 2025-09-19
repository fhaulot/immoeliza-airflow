import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisPreprocessing: 

    def __init__(self):
        pass

    def read_csv(self, path):
        """
        Reading the scraped Immovlan data csv file
        """
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        print(df.info())
        return df
    
    def clean_data(self, df):
        """
        Initial data cleaning for analysis - remove obvious outliers and invalid data
        """
        logger.info("Starting initial data cleaning...")
        initial_rows = len(df)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['price', 'postCode'])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing price or postal code")
        
        # Remove unrealistic prices (< 10k or > 5M EUR)
        df = df[(df['price'] >= 10000) & (df['price'] <= 5000000)]
        logger.info(f"Removed price outliers, remaining rows: {len(df)}")
        
        # Remove unrealistic surfaces (< 10m² or > 1000m²)
        df = df[(df['habitableSurface'].isna()) | 
                ((df['habitableSurface'] >= 10) & (df['habitableSurface'] <= 1000))]
        logger.info(f"Removed surface outliers, remaining rows: {len(df)}")
        
        # Remove unrealistic bedroom counts (0-20)
        df = df[(df['bedroomCount'].isna()) | 
                ((df['bedroomCount'] >= 0) & (df['bedroomCount'] <= 20))]
        logger.info(f"Removed bedroom outliers, remaining rows: {len(df)}")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values appropriately for analysis
        """
        logger.info("Handling missing values...")
        
        # Fill missing bedroom counts with median by property type
        for prop_type in df['type'].unique():
            mask = (df['type'] == prop_type) & df['bedroomCount'].isna()
            median_bedrooms = df[df['type'] == prop_type]['bedroomCount'].median()
            df.loc[mask, 'bedroomCount'] = median_bedrooms
        
        # Fill missing habitable surface with median by type and bedrooms
        for prop_type in df['type'].unique():
            for bedrooms in df['bedroomCount'].unique():
                if pd.isna(bedrooms):
                    continue
                mask = ((df['type'] == prop_type) & 
                       (df['bedroomCount'] == bedrooms) & 
                       df['habitableSurface'].isna())
                median_surface = df[(df['type'] == prop_type) & 
                                  (df['bedroomCount'] == bedrooms)]['habitableSurface'].median()
                if not pd.isna(median_surface):
                    df.loc[mask, 'habitableSurface'] = median_surface
        
        # Fill remaining missing surfaces with overall median by type
        for prop_type in df['type'].unique():
            mask = (df['type'] == prop_type) & df['habitableSurface'].isna()
            median_surface = df[df['type'] == prop_type]['habitableSurface'].median()
            df.loc[mask, 'habitableSurface'] = median_surface
        
        # Fill missing building condition with 'UNKNOWN'
        df['buildingCondition'] = df['buildingCondition'].fillna('UNKNOWN')
        
        # Fill missing EPC score with 'UNKNOWN'
        df['epcScore'] = df['epcScore'].fillna('UNKNOWN')
        
        # Fill missing locality with postal code
        df['locality'] = df['locality'].fillna(df['postCode'].astype(str))
        df['MunicipalityCleanName'] = df['MunicipalityCleanName'].fillna(df['locality'])
        
        return df
    
    def ordinal_building_condition(self, df):
        """
        Perform ordinal encoding for building condition
        """
        condition_order = {
            "AS_NEW": 1, 
            "GOOD": 2, 
            "JUST_RENOVATED": 3,
            "TO_RENOVATE": 4, 
            "TO_BE_DONE_UP": 5, 
            "TO_RESTORE": 6,
            "UNKNOWN": 3  # Assign middle value for unknown
        }
        df['encoded_condition'] = df['buildingCondition'].map(condition_order)
        # Fill any remaining NaN with middle value
        df['encoded_condition'] = df['encoded_condition'].fillna(3)
        logger.info("Applied ordinal encoding to building condition")
        return df
    
    def ordinal_epc_score(self, df):
        """
        Perform ordinal encoding for the column of epc score using a mapping
        """
        epc_order = {
            "A": 1, "B": 2, "C": 3, "D": 4, 
            "E": 5, "F": 6, "G": 7, "UNKNOWN": 4  # Middle value
        }
        df['encoded_epc'] = df['epcScore'].map(epc_order)
        # Fill any remaining NaN with middle value
        df['encoded_epc'] = df['encoded_epc'].fillna(4)
        logger.info("Applied ordinal encoding to EPC score")
        return df
    
    def categorical_encode(self, df):
        """
        Perform categorical encoding on specified columns for analysis.
        """
        # Select categorical columns for encoding
        categorical_columns = []
        
        if 'province' in df.columns:
            categorical_columns.append('province')
        if 'subtype' in df.columns:
            categorical_columns.append('subtype')
        if 'type' in df.columns:
            categorical_columns.append('type')
        if 'region' in df.columns:
            categorical_columns.append('region')
            
        if categorical_columns:
            df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            
            # Convert boolean columns to int
            bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
            df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
            
            logger.info(f"Applied one-hot encoding to: {categorical_columns}")
            return df_encoded
        else:
            logger.warning("No categorical columns found for encoding")
            return df
    
    def add_derived_features(self, df):
        """
        Add derived features for better analysis
        """
        logger.info("Adding derived features...")
        
        # Recalculate price per square meter if needed
        if 'habitableSurface' in df.columns and 'price' in df.columns:
            df['price_per_sqm'] = df['price'] / df['habitableSurface']
            df['price_per_sqm'] = df['price_per_sqm'].replace([np.inf, -np.inf], np.nan)
        
        # Add price categories
        if 'price' in df.columns:
            df['price_category'] = pd.cut(df['price'], 
                                        bins=[0, 200000, 400000, 600000, np.inf], 
                                        labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Add surface categories
        if 'habitableSurface' in df.columns:
            df['surface_category'] = pd.cut(df['habitableSurface'], 
                                          bins=[0, 75, 150, 250, np.inf], 
                                          labels=['Small', 'Medium', 'Large', 'XLarge'])
        
        return df
    
    def prepare_for_analysis(self, df):
        """
        Final preparation steps for analysis
        """
        logger.info("Preparing data for analysis...")
        
        # Columns to keep for analysis (don't drop too much for exploratory analysis)
        analysis_columns = [
            'price', 'habitableSurface', 'bedroomCount', 'postCode',
            'hasGarden', 'gardenSurface', 'hasTerrace', 'hasParking',
            'encoded_condition', 'encoded_epc', 'price_per_sqm'
        ]
        
        # Add encoded categorical columns
        categorical_encoded_cols = [col for col in df.columns if any(
            prefix in col for prefix in ['province_', 'subtype_', 'type_', 'region_']
        )]
        analysis_columns.extend(categorical_encoded_cols)
        
        # Add derived feature columns
        derived_cols = ['price_category', 'surface_category']
        for col in derived_cols:
            if col in df.columns:
                analysis_columns.append(col)
        
        # Filter to existing columns
        existing_columns = [col for col in analysis_columns if col in df.columns]
        df_analysis = df[existing_columns].copy()
        
        logger.info(f"Prepared analysis dataset with {len(existing_columns)} columns")
        logger.info(f"Final shape: {df_analysis.shape}")
        
        return df_analysis
    
    def full_preprocessing_pipeline(self, input_path, output_path):
        """
        Run the complete preprocessing pipeline for analysis
        """
        logger.info("=== Starting Analysis Preprocessing Pipeline ===")
        
        # Load data
        df = self.read_csv(input_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Apply ordinal encoding
        df = self.ordinal_building_condition(df)
        df = self.ordinal_epc_score(df)
        
        # Apply categorical encoding
        df = self.categorical_encode(df)
        
        # Prepare final dataset
        df_final = self.prepare_for_analysis(df)
        
        # Save processed data
        df_final.to_csv(output_path, index=False)
        logger.info(f"Analysis-ready data saved to: {output_path}")
        
        return df_final
    
