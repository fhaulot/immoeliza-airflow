FROM apache/airflow:2.10.3

USER root
# Install docker-cli to allow checking docker status if needed (optional but helpful)
# But mainly we need the provider
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install apache-airflow-providers-docker
