# 🏠 Immoeliza-Airflow

Belgian real estate price prediction pipeline using Immovlan data, with Apache Airflow orchestration.

## 📋 Description

This project implements a complete machine learning pipeline to predict real estate prices in Belgium using scraped data from Immovlan. The pipeline is harmonized and ready for orchestration with Apache Airflow.

## 🗂️ Project Structure

```
📦 immoeliza-airflow/
├── 📁 analyse/                    # Preprocessing for exploratory analysis
│   ├── Preprocessing.py           # AnalysisPreprocessing class
│   ├── cleaned_data.csv           # Cleaned data
│   └── processed_for_analysis.csv # Data ready for analysis
├── 📁 deployment/                 # API deployment
│   ├── app.py                     # Harmonized FastAPI
│   └── Dockerfile.txt             # Docker configuration
├── 📁 model/                      # Machine Learning
│   ├── Preprocessing.py           # ModelPreprocessing class (ML)
│   ├── pipeline.py                # ModelPipeline class (training)
│   ├── 📁 processed_data/         # ML data + scaler + features
│   │   ├── feature_columns.csv    # Saved feature columns
│   │   ├── scaler.pkl             # Saved StandardScaler
│   │   ├── X_train.csv           # Training features
│   │   ├── X_test.csv            # Test features
│   │   ├── y_train.csv           # Training targets
│   │   └── y_test.csv            # Test targets
│   └── 📁 trained_models/         # Trained models
│       ├── best_model.pkl         # Best model (Random Forest)
│       ├── model_metadata.csv     # Model metadata
│       ├── detailed_results.csv   # Detailed results
│       └── 📁 all_models/         # All trained models
│           ├── random_forest.pkl
│           ├── linear_regression.pkl
│           ├── ridge_regression.pkl
│           ├── polynomial_regression.pkl
│           └── gradient_boosting.pkl
├── 📁 scrapper/                   # Data extraction
│   ├── improved_scraping.py       # Optimized extraction functions
│   ├── main_scraper.py           # Main scraping script
│   ├── immovlan.py               # Original Immovlan script
│   └── scrapping.py              # Base scraping script
├── immovlan_sales_urls.txt       # Immovlan source URLs
├── immovlan_single_listing.csv   # Data example
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/fhaulot/immoeliza-airflow.git
cd immoeliza-airflow
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🔄 Data Pipeline

The pipeline consists of 5 main steps:

### 1. 📊 Scraping (scrapper/)
- **Entry point**: `scrapper.main_scraper.full_preprocessing_pipeline()`
- **Input**: `immovlan_sales_urls.txt`
- **Output**: `immovlan_scraped_data.csv`
- **Description**: Extracts property data from Immovlan

### 2. 🔍 Analysis Preprocessing (analyse/)
- **Entry point**: `analyse.Preprocessing.AnalysisPreprocessing().full_preprocessing_pipeline()`
- **Input**: `immovlan_scraped_data.csv`
- **Output**: `analyse/processed_for_analysis.csv`
- **Description**: Cleans and prepares data for exploratory analysis

### 3. 🤖 ML Preprocessing (model/)
- **Entry point**: `model.Preprocessing.ModelPreprocessing().full_preprocessing_pipeline()`
- **Input**: `immovlan_scraped_data.csv`
- **Output**: `model/processed_data/` (train/test splits + scaler + features)
- **Description**: Prepares data for ML training

### 4. 🎯 Training (model/)
- **Entry point**: `model.pipeline.ModelPipeline().full_training_pipeline()`
- **Input**: `model/processed_data/`
- **Output**: `model/trained_models/`
- **Description**: Trains and compares multiple ML models

### 5. 🌐 Deployment (deployment/)
- **Entry point**: FastAPI on `deployment.app:app`
- **Input**: Trained models
- **Output**: REST API for predictions
- **Description**: Deploys the prediction API

## 🔧 Usage

### Manual scraping
```python
from scrapper.main_scraper import full_preprocessing_pipeline
full_preprocessing_pipeline(max_properties=100)
```

### Preprocessing for analysis
```python
from analyse.Preprocessing import AnalysisPreprocessing
processor = AnalysisPreprocessing()
processor.full_preprocessing_pipeline()
```

### Model training
```python
from model.pipeline import ModelPipeline
pipeline = ModelPipeline()
pipeline.full_training_pipeline()
```

### Launch API
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

## 📊 API Endpoints

### GET /
API homepage

### POST /predict
Real estate price prediction

**Request example:**
```json
{
  "habitableSurface": 120,
  "bedroomCount": 3,
  "postCode": 1000,
  "type": "APARTMENT",
  "subtype": "APARTMENT", 
  "province": "Brussels",
  "region": "Brussels",
  "buildingCondition": "Good",
  "epcScore": "B",
  "hasGarden": 1,
  "hasTerrace": 0,
  "hasParking": 1
}
```

**Response example:**
```json
{
  "predicted_price": 282770.60,
  "currency": "EUR",
  "confidence_score": 0.318,
  "model_name": "Random Forest",
  "status": "success",
  "timestamp": "2025-09-19T16:34:09.170716"
}
```

## 🎛️ Airflow Configuration

The project is structured for easy integration with Apache Airflow. Each pipeline step can be transformed into an Airflow task:

```python
# Airflow DAG example
from airflow import DAG
from airflow.operators.python import PythonOperator

def scraping_task():
    from scrapper.main_scraper import full_preprocessing_pipeline
    return full_preprocessing_pipeline(max_properties=100)

def analysis_preprocessing_task():
    from analyse.Preprocessing import AnalysisPreprocessing
    processor = AnalysisPreprocessing()
    return processor.full_preprocessing_pipeline()

# ... define other tasks
```

## 📈 Model Performance

- **Selected model**: Random Forest
- **R² Score**: 0.318
- **MAE**: €80,291
- **Training data**: Belgian Immovlan properties

## 🛠️ Technologies Used

- **Python 3.12**: Main language
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **FastAPI**: REST API
- **BeautifulSoup**: Web scraping
- **Apache Airflow**: Orchestration (to be implemented)

## 📝 Authors

- Floriane Haulot (fhaulot)
- AI Assistant (harmonization and structuring)

## 📄 License

This project is under [to be defined] license.
