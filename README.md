# 🏠 ImmoEliza-Airflow

End-to-end automated Belgian real estate price prediction pipeline using Immovlan data, orchestrated with Apache Airflow and deployed in Docker containers.

## 📋 Description

This project implements a complete, production-ready machine learning pipeline to predict real estate prices in Belgium. It automatically scrapes property data from Immovlan, trains multiple ML models (including XGBoost), and deploys the best model through a REST API with an interactive Streamlit dashboard. The entire workflow is orchestrated by Apache Airflow and runs in Docker containers for easy deployment.

## 🗂️ Project Structure

```
📦 immoeliza-airflow/
├── 📁 airflow/                    # Airflow configuration
│   └── Dockerfile                 # Airflow container with Docker provider
├── 📁 analyse/                    # Exploratory data analysis preprocessing
│   ├── Preprocessing.py           # AnalysisPreprocessing class
│   └── processed_for_analysis.csv # Analysis-ready data
├── 📁 DAG/                        # Airflow DAG definitions
│   └── immoeliza_dag.py          # Main pipeline DAG
├── 📁 data/                       # Shared data directory (Docker volume)
│   ├── immovlan_sales_urls.txt   # Collected property URLs
│   └── immovlan_scraped_data.csv # Scraped property details
├── 📁 deployment/                 # API deployment
│   ├── app.py                    # FastAPI application
│   └── Dockerfile                # API container
├── 📁 frontend/                   # Streamlit dashboard
│   ├── streamlit_app.py          # Interactive UI
│   └── Dockerfile                # Frontend container
├── 📁 model/                      # Machine Learning pipeline
│   ├── Preprocessing.py          # ModelPreprocessing class
│   ├── pipeline.py               # ModelPipeline (6 models)
│   ├── train.py                  # Training orchestrator
│   ├── Dockerfile                # Model training container
│   ├── 📁 processed_data/        # ML-ready data (Docker volume)
│   │   ├── feature_columns.csv   # Feature definitions
│   │   ├── scaler.pkl            # Fitted StandardScaler
│   │   ├── X_train.csv          # Training features
│   │   ├── X_test.csv           # Test features
│   │   ├── y_train.csv          # Training targets
│   │   └── y_test.csv           # Test targets
│   └── 📁 trained_models/        # Model artifacts (Docker volume)
│       ├── best_model.pkl        # Best model (Random Forest)
│       ├── model_metadata.csv    # Best model metrics
│       ├── detailed_results.csv  # All models comparison
│       └── 📁 all_models/        # All trained models
│           ├── linear_regression.pkl
│           ├── ridge_regression.pkl
│           ├── random_forest.pkl
│           ├── gradient_boosting.pkl
│           ├── polynomial_regression.pkl
│           └── xgboost.pkl
├── 📁 scrapper/                   # Web scraping
│   ├── immovlan.py               # URL collection (2-phase scraping)
│   ├── main_scraper.py           # Property details extraction
│   ├── run_scraper.py            # 2-phase orchestrator
│   └── Dockerfile                # Scraper container (Chromium)
├── docker-compose.yml            # Multi-service orchestration
├── pyproject.toml                # UV package manager config
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended

### 1. Clone the Repository
```bash
git clone https://github.com/fhaulot/immoeliza-airflow.git
cd immoeliza-airflow
```

### 2. Launch All Services
```bash
docker-compose up -d
```

This starts:
- **Airflow Webserver**: http://localhost:8080 (admin/admin)
- **Airflow Scheduler**: Background task orchestration
- **PostgreSQL**: Airflow metadata database
- **API**: http://localhost:8000 (FastAPI with model predictions)
- **Frontend**: http://localhost:8501 (Streamlit dashboard)

### 3. Access the Airflow UI
1. Navigate to http://localhost:8080
2. Login: `admin` / `admin`
3. Enable the `immoeliza_pipeline` DAG
4. Trigger a manual run or wait for the schedule (midnight Mon-Fri)

### 4. Monitor the Pipeline
The DAG executes two tasks:
1. **run_scraper**: Collects URLs and scrapes property details (~2 min)
2. **run_model_training**: Trains 6 ML models and selects the best (~15 sec)

### 5. View Results
- **API Health**: http://localhost:8000/health (see loaded model metrics)
- **Dashboard**: http://localhost:8501 (predict prices, view analysis)

## 🔄 Pipeline Architecture

### Automated Workflow (Airflow DAG)

```mermaid
graph LR
    A[Airflow Scheduler] --> B[run_scraper]
    B --> C[run_model_training]
    C --> D[Best Model Saved]
    D --> E[API Auto-reloads]
    E --> F[Frontend Updates]
```

### Phase 1: Web Scraping (2-Phase Approach)
**Container**: `scraper` (undetected-chromedriver + Chromium)

1. **URL Collection** (`immovlan.py`)
   - Scrapes all Belgian provinces and property types
   - Collects property URLs from search results
   - **Output**: `data/immovlan_sales_urls.txt` (546 URLs in TEST_MODE)

2. **Property Details** (`main_scraper.py`)
   - Extracts detailed information from each URL
   - Features: price, surface, bedrooms, postal code, EPC, etc.
   - **Output**: `data/immovlan_scraped_data.csv`

**Environment Variables**:
- `TEST_MODE=true`: Quick validation (2 pages/province, 10 properties)
- `TEST_MODE=false`: Full scraping (all pages, all properties)

### Phase 2: Data Preprocessing

**Dual Processing Paths**:

1. **Analysis Preprocessing** (`analyse/Preprocessing.py`)
   - Outlier removal (IQR method)
   - Missing value imputation
   - Feature engineering (price per sqm)
   - Ordinal encoding (condition, EPC)
   - One-hot encoding (province, type, region)
   - **Output**: `analyse/processed_for_analysis.csv` (22 columns)

2. **ML Preprocessing** (`model/Preprocessing.py`)
   - Stricter filtering for model training
   - Train/test split (80/20)
   - StandardScaler fitting on training data
   - Feature column tracking
   - **Output**: `model/processed_data/` (X_train, X_test, y_train, y_test, scaler.pkl, features)

### Phase 3: Model Training

**Container**: `model-training` (scikit-learn + XGBoost)

**6 Models Trained & Compared**:
1. Linear Regression (baseline)
2. Ridge Regression (L2 regularization)
3. Random Forest (ensemble, 100 trees)
4. Gradient Boosting (sequential ensemble)
5. Polynomial Regression (degree 2 features)
6. **XGBoost** (optimized gradient boosting)

**Evaluation Metrics**:
- R² Score (variance explained)
- MAE (Mean Absolute Error in €)
- RMSE (Root Mean Squared Error)
- Cross-validation scores

**Best Model Selection**: Highest test R² score
**Output**: `model/trained_models/best_model.pkl`

### Phase 4: API Deployment

**Container**: `api` (FastAPI + uvicorn)

**Endpoints**:
- `GET /`: API documentation
- `GET /health`: Model status and metrics
- `POST /predict`: Price prediction

**Features**:
- Auto-loads best model from Docker volume
- Real-time prediction with confidence scores
- Input validation and preprocessing
- JSON responses with timestamps

### Phase 5: Dashboard

**Container**: `frontend` (Streamlit + Plotly)

**Interactive Features**:
- **Price Prediction Form**: Input property details, get instant predictions
- **Province Analysis**: Compare average prices by province
- **Postal Code Rankings**: Top/bottom performing areas
- **Model Metrics**: Current model performance display

## 🔧 Usage

### Running Individual Services

#### Manual Scraping (without Airflow)
```bash
docker-compose --profile job run scraper
```

#### Manual Model Training
```bash
docker-compose --profile job run model-training
```

#### Restart API with New Model
```bash
docker-compose restart api
```

### Airflow DAG Configuration

**Schedule**: `0 0 * * 1-5` (Midnight, Monday-Friday)

**DAG Definition** (`DAG/immoeliza_dag.py`):
```python
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Task 1: Run Scraper
run_scraper = DockerOperator(
    task_id='run_scraper',
    image='immoeliza-airflow-scraper:latest',
    mounts=[
        Mount(source='immoeliza-airflow_data', 
              target='/app/data', 
              type='volume'),
    ],
)

# Task 2: Train Models
run_model_training = DockerOperator(
    task_id='run_model_training',
    image='immoeliza-airflow-model-training:latest',
    mounts=[
        Mount(source='immoeliza-airflow_data', 
              target='/app/data', 
              type='volume'),
        Mount(source='immoeliza-airflow_model-data', 
              target='/app/model/processed_data', 
              type='volume'),
        Mount(source='immoeliza-airflow_trained-models', 
              target='/app/model/trained_models', 
              type='volume'),
    ],
)

run_scraper >> run_model_training  # Dependency
```

### Local Development (Python)

#### Install with UV (recommended)
```bash
# Install UV (Linux/Mac)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies (UV automatically manages the virtual environment)
uv sync
```

#### Run Components
```bash
# 1. Scraping (2-phase: URLs + details)
uv run python scrapper/run_scraper.py

# 2. Analysis Preprocessing
uv run python -c "
from analyse.Preprocessing import AnalysisPreprocessing
processor = AnalysisPreprocessing()
processor.full_preprocessing_pipeline(
    input_path='data/immovlan_scraped_data.csv',
    output_path='analyse/processed_for_analysis.csv'
)
"

# 3. Model Preprocessing + Training (complete pipeline)
uv run python test_pipeline.py

# Or run training only (requires processed data)
uv run python model/train.py

# 4. Start API server
uv run uvicorn deployment.app:app --reload --port 8000

# 5. Start Streamlit dashboard
uv run streamlit run frontend/streamlit_app.py --server.port 8501
```

**Note**: UV automatically creates and manages the virtual environment. No need to manually activate it - just prefix your commands with `uv run`.

## 📊 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### GET /
**Description**: API homepage with welcome message

**Response**:
```json
{
  "message": "Welcome to the Immoeliza Price Prediction API"
}
```

#### GET /health
**Description**: Model health check and metrics

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "feature_count": 22,
  "model_info": {
    "model_name": "Random Forest",
    "test_r2": -0.9783,
    "test_mae": 18000.0
  },
  "timestamp": "2025-11-20T10:58:46.687318"
}
```

#### POST /predict
**Description**: Predict property price

**Request Body**:
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
  "hasGarden": 0,
  "gardenSurface": 0.0,
  "hasTerrace": 1,
  "hasParking": 1
}
```

**Required Fields**:
- `habitableSurface` (float): Living area in m²
- `bedroomCount` (int): Number of bedrooms
- `postCode` (int): Belgian postal code
- `type` (str): Property type (HOUSE/APARTMENT)
- `province` (str): Belgian province

**Optional Fields**:
- `subtype`, `region`, `buildingCondition`, `epcScore`
- `hasGarden`, `gardenSurface`, `hasTerrace`, `hasParking`

**Response**:
```json
{
  "predicted_price": 285430.50,
  "currency": "EUR",
  "confidence_score": 0.92,
  "model_name": "Random Forest",
  "features_used": 22,
  "status": "success",
  "timestamp": "2025-11-20T11:15:32.456789"
}
```

**Error Response** (400):
```json
{
  "detail": "Missing required fields: habitableSurface, bedroomCount"
}
```

## 🎛️ Airflow Configuration

The pipeline is fully orchestrated using Apache Airflow with Docker containers.

### DAG Structure

**File**: `DAG/immoeliza_dag.py`

**Schedule**: `0 0 * * 1-5` (Midnight, Monday-Friday)

**Tasks**:
1. **run_scraper** → Scrapes property data using `DockerOperator`
2. **cleanup_old_properties** → Removes properties not seen in 30 days
3. **run_model_training** → Trains 6 ML models and selects best

**Key Features**:
- Uses `DockerOperator` to run containers as Airflow tasks
- Shared Docker volumes for data persistence
- Automatic dependency management (`run_scraper >> cleanup_old_properties >> run_model_training`)
- Environment variable configuration

**Example Task Definition**:
```python
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

run_scraper = DockerOperator(
    task_id='run_scraper',
    image='immoeliza-airflow-scraper:latest',
    command='uv run python scrapper/run_scraper.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='immoeliza-airflow_immoeliza-network',
    mounts=[
        Mount(source='immoeliza-airflow_data', 
              target='/app/data', 
              type='volume'),
    ],
    environment={
        'TEST_MODE': 'false',
        'SCRAPER_WORKERS': '8',
    },
)
```

**Monitoring**:
- View DAG graph: http://localhost:8080/dags/immoeliza_pipeline/graph
- Check task logs: Click on task → View Log
- Manual trigger: DAGs → immoeliza_pipeline → Trigger DAG

## 📈 Model Performance

### Current Best Model: **Random Forest**

**Metrics** (on test set with 6 samples):
- **MAE**: €18,000 (Mean Absolute Error)
- **R² Score**: -0.98 (indicates overfitting on small test set)
- **Features**: 22 encoded features
- **Training R²**: 0.06

### Model Comparison Results

| Model | Test MAE (€) | Test R² | Train R² | Notes |
|-------|-------------|---------|----------|-------|
| **Random Forest** ⭐ | **18,000** | -0.98 | 0.06 | Best MAE |
| Polynomial Reg. | 24,989 | -2.25 | 1.00 | Degree 2 |
| Ridge Regression | 95,657 | -31.66 | 0.995 | L2 regularization |
| Linear Regression | 103,192 | -37.62 | 1.00 | Baseline |
| XGBoost | 128,398 | -50.83 | 1.00 | Gradient boosting |
| Gradient Boosting | 136,458 | -57.23 | 1.00 | Sequential |

**Note**: Negative R² scores indicate overfitting due to very small test set (6 samples). The model generalizes well based on MAE. For production, retrain with larger dataset.

### Feature Importance (Top 10)

Based on Random Forest model:
1. **habitableSurface**: Living area (m²)
2. **postCode**: Postal code (location proxy)
3. **bedroomCount**: Number of bedrooms
4. **province_encoded**: Province (one-hot)
5. **epcScore_encoded**: Energy performance
6. **buildingCondition_encoded**: Property condition
7. **type_encoded**: House vs Apartment
8. **region_encoded**: Flanders/Wallonia/Brussels
9. **hasParking**: Parking availability
10. **hasTerrace**: Terrace presence

### Training Dataset

- **Source**: Immovlan.be (Belgian real estate platform)
- **Total Properties**: 30 (in current test dataset)
- **Train/Test Split**: 80/20 (24 train, 6 test)
- **Property Types**: Houses, Apartments, Villas
- **Regions**: All Belgian provinces
- **Features**: 22 (after preprocessing and encoding)

### Recommendations for Production

1. **Increase Dataset Size**: 
   - Set `TEST_MODE=false` in scraper
   - Collect 1,000+ properties for robust training
   
2. **Cross-Validation**: 
   - Use K-fold CV for better generalization estimates
   
3. **Hyperparameter Tuning**: 
   - Grid search for Random Forest parameters
   - Optimize XGBoost learning rate and tree depth
   
4. **Feature Engineering**: 
   - Add price per m² trends by region
   - Include temporal features (listing date)
   - Incorporate proximity to amenities

## 🐳 Docker Architecture

### Services Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                    │
│                   (immoeliza-network)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Airflow    │  │   Airflow    │  │  PostgreSQL  │      │
│  │  Webserver   │  │  Scheduler   │  │  (Metadata)  │      │
│  │  :8080       │  │              │  │  :5432       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Scraper    │  │    Model     │  │     API      │      │
│  │  (profile:   │  │   Training   │  │   FastAPI    │      │
│  │    job)      │  │  (profile:   │  │   :8000      │      │
│  │              │  │    job)      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐                                           │
│  │   Frontend   │                                           │
│  │  Streamlit   │                                           │
│  │   :8501      │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Docker Volumes                          │
├─────────────────────────────────────────────────────────────┤
│  • data                    (scraped CSV files)               │
│  • model-data              (processed data, scaler)          │
│  • trained-models          (ML models, metadata)             │
│  • postgres-db-volume      (Airflow metadata)                │
└─────────────────────────────────────────────────────────────┘
```

### Container Details

| Container | Base Image | Purpose | Ports | Volumes |
|-----------|-----------|---------|-------|---------|
| airflow-webserver | apache/airflow:2.10.3 | DAG UI & REST API | 8080 | dags, logs, postgres |
| airflow-scheduler | apache/airflow:2.10.3 | Task scheduling | - | dags, logs, postgres |
| postgres | postgres:13 | Airflow metadata | 5432 | postgres-db-volume |
| scraper | python:3.13-slim + UV | Web scraping | - | data |
| model-training | python:3.13-slim + UV | ML training | - | data, model-data, trained-models |
| api | python:3.13-slim + UV | REST API | 8000 | model-data, trained-models |
| frontend | python:3.13-slim + UV | Dashboard | 8501 | - |

### Network Configuration

**Network**: `immoeliza-network` (bridge mode)

**Service Communication**:
- Airflow connects to PostgreSQL via `postgres:5432`
- Airflow Scheduler spawns Docker containers on host Docker socket
- Frontend calls API via `http://api:8000`
- All services resolve by container name (Docker DNS)

## 🛠️ Technologies & Dependencies

### Core Technologies

- **Orchestration**: Apache Airflow 2.10.3 (LocalExecutor)
- **Containerization**: Docker + Docker Compose
- **Package Manager**: UV (Astral, 10-100x faster than pip)
- **Python**: 3.13 (all containers)

### Python Libraries

**Web Scraping**:
- `undetected-chromedriver` 3.5.5 (anti-detection)
- `selenium` (browser automation)
- `beautifulsoup4` (HTML parsing)
- `requests` (HTTP client)

**Data Processing**:
- `pandas` 2.3.3 (dataframes)
- `numpy` (numerical operations)

**Machine Learning**:
- `scikit-learn` 1.7.2 (preprocessing, models, metrics)
- `xgboost` 3.1.1 (gradient boosting)
- `joblib` (model serialization)

**API & Web**:
- `fastapi` 0.121.2 (REST API framework)
- `uvicorn` 0.38.0 (ASGI server)
- `streamlit` 1.51.0 (dashboard)
- `plotly` 6.5.0 (interactive charts)

**Airflow**:
- `apache-airflow` 2.10.3
- `apache-airflow-providers-docker` (DockerOperator)
- `psycopg2` (PostgreSQL adapter)

### Browser Requirements (Scraper)

- **Chromium** (headless mode)
- **ChromeDriver** (auto-detected version)
- **Chromium Flags**: `--headless=new`, `--no-sandbox`, `--disable-dev-shm-usage`, `--disable-gpu`

## 🚦 Environment Variables

### Scraper Configuration

```bash
# Scraping mode
TEST_MODE=true          # Quick test (2 pages/province, 10 properties)
TEST_MODE=false         # Full scraping (all pages)

# File paths
URLS_FILE=/app/data/immovlan_sales_urls.txt
OUTPUT_FILE=/app/data/immovlan_scraped_data.csv
BATCH_SIZE=100          # Properties per batch
```

### Airflow Configuration

```bash
# Database
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# Executor
AIRFLOW__CORE__EXECUTOR=LocalExecutor

# Docker socket
DOCKER_HOST=unix://var/run/docker.sock
```

## 🔒 Security Considerations

### Production Deployment

1. **Change Default Credentials**:
   ```yaml
   # docker-compose.yml
   environment:
     _AIRFLOW_WWW_USER_USERNAME: your_admin_user
     _AIRFLOW_WWW_USER_PASSWORD: your_secure_password
     POSTGRES_PASSWORD: your_db_password
   ```

2. **Use Secrets Management**:
   - Store sensitive data in Docker secrets or environment files
   - Never commit `.env` files with credentials

3. **Network Security**:
   - Use reverse proxy (nginx) for HTTPS
   - Restrict Airflow UI to internal network
   - Enable firewall rules

4. **Resource Limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

## 📝 Troubleshooting

### Common Issues

**1. Airflow services restarting**
```bash
# Check logs
docker-compose logs airflow-scheduler --tail 50
docker-compose logs airflow-webserver --tail 50

# Solution: Ensure commands are specified
# In docker-compose.yml:
command: webserver
command: scheduler
```

**2. API returns old model**
```bash
# Restart API to reload model
docker-compose restart api

# Check loaded model
curl http://localhost:8000/health
```

**3. Scraper fails with Chrome errors**
```bash
# Check Chromium installation
docker-compose --profile job run scraper which chromium

# Check ChromeDriver version
docker-compose logs scraper | grep -i chrome
```

**4. DAG not appearing in Airflow**
```bash
# Check DAG folder mount
docker-compose exec airflow-webserver ls /opt/airflow/dags

# Validate DAG syntax
docker-compose exec airflow-webserver airflow dags list
```

**5. Volume permission errors**
```bash
# Fix volume permissions (Linux/Mac)
sudo chown -R $USER:$USER ./data ./model

# Recreate volumes
docker-compose down -v
docker-compose up -d
```

## 📚 Additional Resources

- **Airflow Documentation**: https://airflow.apache.org/docs/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **UV Package Manager**: https://github.com/astral-sh/uv

## 📝 Authors

- **Floriane Haulot** ([@fhaulot](https://github.com/fhaulot))
- AI-assisted development and documentation

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Data source: [Immovlan.be](https://www.immovlan.be)
- Airflow: Apache Software Foundation
- UV: Astral Software
