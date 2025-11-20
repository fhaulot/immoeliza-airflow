from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from docker.types import Mount

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

    # Define the dependency
    run_scraper >> run_model_training
