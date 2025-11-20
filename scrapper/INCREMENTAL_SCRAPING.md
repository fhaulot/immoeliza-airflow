# Incremental Scraping & Data Retention

## Overview

The scraper now implements **incremental scraping** with automatic metrics tracking and data retention policies.

## Key Features

### 1. Incremental Scraping
- **Detects existing data**: Loads previous scraping results
- **Identifies changes**: Compares current URLs with existing ones
- **Scrapes only new properties**: Avoids re-scraping existing listings
- **Updates timestamps**: Tracks when each property was last seen

### 2. Metrics Tracking

Each scraping run generates a metrics file: `immovlan_scraped_data_metrics.json`

```json
{
  "new_properties": 15,
  "updated_properties": 234,
  "removed_properties": 8,
  "total_properties": 241,
  "timestamp": "2025-11-20T12:34:56.789"
}
```

**Metrics breakdown:**
- `new_properties`: New listings found (scraped in this run)
- `updated_properties`: Existing listings still active (timestamp updated)
- `removed_properties`: Listings no longer available
- `total_properties`: Total properties in database after this run
- `timestamp`: When the scraping run completed

### 3. Data Retention Policy

The `cleanup_old_properties.py` script removes stale listings:

**Default behavior:**
- Properties not seen in **30 days** are removed
- Creates automatic backup before deletion
- Generates cleanup metrics file

**Configuration via environment variables:**
```bash
DATA_FILE=/data/immovlan_scraped_data.csv
CLEANUP_DAYS_THRESHOLD=30
DRY_RUN=false  # Set to true to preview without deleting
```

**Manual cleanup:**
```bash
# Preview what would be removed (dry run)
docker-compose run --rm scraper uv run python scrapper/cleanup_old_properties.py \
  -e DRY_RUN=true

# Actually remove old properties
docker-compose run --rm scraper uv run python scrapper/cleanup_old_properties.py
```

### 4. DAG Integration

The Airflow DAG now runs 3 tasks in sequence:

```
run_scraper → cleanup_old_properties → run_model_training
```

**Task flow:**
1. **Scraper**: Collects URLs, scrapes new properties, updates timestamps
2. **Cleanup**: Removes properties not seen in 30 days
3. **Model Training**: Trains on the cleaned, up-to-date dataset

## How It Works

### First Run (No existing data)
```
URLs found: 500
Existing properties: 0
New URLs to scrape: 500
→ Scrapes all 500 properties
→ Saves with last_seen timestamp
```

### Second Run (with existing data)
```
URLs found: 520
Existing properties: 500
New URLs to scrape: 20
Removed URLs: 0
→ Only scrapes 20 new properties
→ Updates last_seen for existing 500
→ Total: 520 properties
```

### Third Run (some properties removed)
```
URLs found: 515
Existing properties: 520
New URLs to scrape: 10
Removed URLs: 15
→ Scrapes 10 new properties
→ Updates last_seen for active 505
→ Total: 515 properties (5 removed from previous run)
```

### After Cleanup (30 days later)
```
Properties before cleanup: 515
Properties not seen in 30 days: 25
Properties after cleanup: 490
→ Creates backup: immovlan_scraped_data_backup_20251220_120000.csv
→ Removes 25 stale listings
```

## Benefits

1. **Efficiency**: Only scrapes new properties (saves time and bandwidth)
2. **Visibility**: Metrics show growth/decline in property listings
3. **Data Quality**: Automatic removal of stale/sold properties
4. **Monitoring**: Track scraping performance over time
5. **Storage**: Prevents database from growing indefinitely

## Monitoring Metrics

Check scraping metrics in Airflow logs:
```bash
# View scraper task logs
docker-compose exec airflow-webserver airflow tasks logs immoeliza_pipeline run_scraper latest

# Check metrics file
docker run --rm -v immoeliza-airflow_data:/data alpine \
  cat /data/immovlan_scraped_data_metrics.json
```

## Advanced Configuration

### Custom Cleanup Schedule

To run cleanup only on Mondays (weekly):
```python
# In DAG file, add trigger rule
cleanup_old_properties = DockerOperator(
    ...
    trigger_rule='all_success',
)
```

### Adjust Retention Period

For 60-day retention:
```yaml
# docker-compose.yml
scraper:
  environment:
    - CLEANUP_DAYS_THRESHOLD=60
```

### Backup Management

Cleanup creates timestamped backups:
```
immovlan_scraped_data_backup_20251220_120000.csv
immovlan_scraped_data_backup_20251227_120000.csv
...
```

Remove old backups manually or with a cron job:
```bash
# Delete backups older than 90 days
find /data -name "*_backup_*.csv" -mtime +90 -delete
```

## Data Schema

The CSV now includes a `last_seen` column:

```csv
url,price,habitableSurface,bedroomCount,last_seen,...
https://immovlan.be/...,350000,120,3,2025-11-20T12:34:56,...
```

This timestamp is updated every time the property is found during scraping.

## Troubleshooting

**Problem**: Metrics file not created
- **Solution**: Ensure scraper has write permissions to `/data` volume

**Problem**: All properties scraped every time (no incremental)
- **Solution**: Check that `immovlan_scraped_data.csv` exists and has `url` column

**Problem**: Cleanup removes too many properties
- **Solution**: Increase `CLEANUP_DAYS_THRESHOLD` or set `DRY_RUN=true` to preview

**Problem**: Want to force full re-scrape
- **Solution**: Delete `immovlan_scraped_data.csv` before running DAG
