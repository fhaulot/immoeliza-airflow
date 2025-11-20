from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from docker.types import Mount
import json
import logging

logger = logging.getLogger(__name__)

def log_scraping_metrics(**context):
    """Read and log scraping metrics to Airflow logs."""
    try:
        # This would need to read from the Docker volume
        # For now, we'll log a placeholder message
        logger.info("=== SCRAPING METRICS ===")
        logger.info("Check scraper logs for detailed metrics")
        logger.info("Metrics file: /data/immovlan_scraped_data_metrics.json")
    except Exception as e:
        logger.error(f"Could not read metrics: {e}")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'immoeliza_pipeline',
    default_args=default_args,
    description='A pipeline to scrape real estate data and train a prediction model',
    schedule_interval='0 0 * * 1-5',  # Run at 00:00 Mon-Fri
    start_date=days_ago(1),
    catchup=False,
    tags=['immoeliza', 'scraping', 'ml'],
) as dag:

    # Task 1: Run the Scraper
    # This runs the scraper container to fetch new data
    run_scraper = DockerOperator(
        task_id='run_scraper',
        image='immoeliza-airflow-scraper:latest',
        api_version='auto',
        auto_remove=True,
        command='uv run python scrapper/run_scraper.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='immoeliza-airflow_immoeliza-network',  # Connect to the same network
        mounts=[
            Mount(source='immoeliza-airflow_data', target='/app/data', type='volume'),
        ],
        environment={
            'TEST_MODE': 'false',
            'URLS_FILE': '/app/data/immovlan_sales_urls.txt',
            'OUTPUT_FILE': '/app/data/immovlan_scraped_data.csv',
            'BATCH_SIZE': '1000',  # Large batch for efficient processing
            'SCRAPER_WORKERS': '8',  # Parallel workers for scraping
        },
        # Increase timeout for long scraping jobs
        execution_timeout=timedelta(hours=2),
        force_pull=False,
    )

    # Task 2: Run Model Training
    # This runs the training container using the data from the scraper
    run_model_training = DockerOperator(
        task_id='run_model_training',
        image='immoeliza-airflow-model-training:latest',
        api_version='auto',
        auto_remove=True,
        command='uv run python model/train.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='immoeliza-airflow_immoeliza-network',
        mounts=[
            Mount(source='immoeliza-airflow_data', target='/app/data', type='volume'),
            Mount(source='immoeliza-airflow_model-data', target='/app/model/processed_data', type='volume'),
            Mount(source='immoeliza-airflow_trained-models', target='/app/model/trained_models', type='volume'),
        ],
        execution_timeout=timedelta(hours=1),
        force_pull=False,
    )

    # Task 3: Cleanup old properties (runs weekly on Monday)
    # Remove properties not seen in 30 days
    cleanup_old_properties = DockerOperator(
        task_id='cleanup_old_properties',
        image='immoeliza-airflow-scraper:latest',
        api_version='auto',
        auto_remove=True,
        command='uv run python scrapper/cleanup_old_properties.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='immoeliza-airflow_immoeliza-network',
        mounts=[
            Mount(source='immoeliza-airflow_data', target='/data', type='volume'),
        ],
        environment={
            'DATA_FILE': '/data/immovlan_scraped_data.csv',
            'CLEANUP_DAYS_THRESHOLD': '30',
            'DRY_RUN': 'false',
        },
        execution_timeout=timedelta(minutes=10),
        force_pull=False,
    )

    # Define the dependencies
    # Scraper -> Cleanup -> Model Training
    run_scraper >> cleanup_old_properties >> run_model_training
