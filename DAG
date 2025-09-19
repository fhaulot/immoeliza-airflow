from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "fhaulot",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="immo_etl_pipeline",
    default_args=default_args,
    description="Orchestration ImmoEliza: scraping -> preprocess -> train -> deploy",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
) as dag:

    workspace = "/opt/airflow/immo_workspace"
    github_user = "fhaulot"

    # 1) ensure workspace
    mk_workspace = BashOperator(
        task_id="make_workspace",
        bash_command=f"mkdir -p {workspace} && chmod 777 {workspace}"
    )

    # 2) clone or pull repos
    clone_repos = BashOperator(
        task_id="clone_repos",
        bash_command=(
            f"cd {workspace} && "
            f"for repo in immoeliza-airflow Immo-Eliza-Scraping Immo-Eliza-data-analysis Immo_Eliza_Regression Immo_Eliza_Deployment; do "
            "if [ -d \"$repo\" ]; then cd $repo && git pull && cd ..; else git clone https://github.com/${GITHUB_USER}/$repo; fi; "
            "done"
        ),
        env={"GITHUB_USER": github_user},
    )

    # 3) scraping
    run_scraper = BashOperator(
        task_id="run_scraper",
        bash_command=(
            f"cd {workspace}/Immo-Eliza-Scraping && "
            # adapt: if you use selenium maybe need display or headless driver set up in environment
            "python3 scraper.py --output immovlan_raw_data_new_v2.csv || true"
        ),
    )

    # 4) preprocessing (data-analysis or regression preprocessing)
    run_preprocessing = BashOperator(
        task_id="run_preprocessing",
        bash_command=(
            f"cd {workspace}/Immo-Eliza-data-analysis && "
            "python3 main.py || true"
        ),
    )

    # 5) train model (regression repo)
    run_training = BashOperator(
        task_id="run_training",
        bash_command=(
            f"cd {workspace}/Immo_Eliza_Regression && "
            "python3 pipeline.py || true"
        ),
    )

    # 6) deploy: build image or kick off deployment scripts
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command=(
            f"cd {workspace}/Immo_Eliza_Deployment && "
            # Option 1: build and run docker (if docker available in env)
            "docker build -t immo_eliza:latest . || true && "
            "docker-compose -f docker-compose.yml up -d || true"
        ),
    )

    # simple ordering
    mk_workspace >> clone_repos >> run_scraper >> run_preprocessing >> run_training >> deploy_model
