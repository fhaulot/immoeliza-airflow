"""
Script to clean up old properties that haven't been seen in X days.
Run this periodically to remove stale listings from the database.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_properties(
    data_file: str = '/data/immovlan_scraped_data.csv',
    days_threshold: int = 30,
    dry_run: bool = False
) -> dict:
    """
    Remove properties that haven't been seen in the last X days.
    
    Args:
        data_file: Path to the CSV file
        days_threshold: Number of days after which a property is considered stale
        dry_run: If True, only show what would be removed without actually removing
    
    Returns:
        dict: Cleanup metrics
    """
    metrics = {
        'total_before': 0,
        'removed': 0,
        'total_after': 0,
        'threshold_days': days_threshold,
        'dry_run': dry_run,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return metrics
    
    # Load data
    df = pd.read_csv(data_file, encoding='utf-8')
    metrics['total_before'] = len(df)
    
    # Check if last_seen column exists
    if 'last_seen' not in df.columns:
        logger.warning("No 'last_seen' column found. Cannot clean up old properties.")
        logger.info("Tip: Run the scraper at least once with the new incremental logic to add this column.")
        return metrics
    
    # Convert last_seen to datetime
    df['last_seen'] = pd.to_datetime(df['last_seen'])
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=days_threshold)
    logger.info(f"Cutoff date: {cutoff_date.isoformat()}")
    
    # Find old properties
    old_properties = df[df['last_seen'] < cutoff_date]
    active_properties = df[df['last_seen'] >= cutoff_date]
    
    metrics['removed'] = len(old_properties)
    metrics['total_after'] = len(active_properties)
    
    logger.info(f"\n=== CLEANUP SUMMARY ===")
    logger.info(f"Total properties before: {metrics['total_before']}")
    logger.info(f"Properties to remove (not seen in {days_threshold} days): {metrics['removed']}")
    logger.info(f"Properties to keep: {metrics['total_after']}")
    
    if metrics['removed'] > 0:
        logger.info(f"\nOldest property last seen: {old_properties['last_seen'].min()}")
        logger.info(f"Sample of properties to remove:")
        logger.info(old_properties[['url', 'last_seen']].head(10).to_string())
    
    # Save cleaned data (unless dry run)
    if not dry_run and metrics['removed'] > 0:
        # Backup original file
        backup_file = data_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(backup_file, index=False, encoding='utf-8')
        logger.info(f"Backup saved to: {backup_file}")
        
        # Save cleaned data
        active_properties.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"Cleaned data saved to: {data_file}")
        
        # Save cleanup metrics
        metrics_file = data_file.replace('.csv', '_cleanup_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Cleanup metrics saved to: {metrics_file}")
    elif dry_run:
        logger.info("\n*** DRY RUN MODE - No changes made ***")
    else:
        logger.info("\nNo properties to remove.")
    
    return metrics

def main():
    """Main entry point with configurable parameters from environment variables."""
    data_file = os.getenv('DATA_FILE', '/data/immovlan_scraped_data.csv')
    days_threshold = int(os.getenv('CLEANUP_DAYS_THRESHOLD', '30'))
    dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    logger.info(f"Starting cleanup with parameters:")
    logger.info(f"  Data file: {data_file}")
    logger.info(f"  Days threshold: {days_threshold}")
    logger.info(f"  Dry run: {dry_run}")
    
    metrics = cleanup_old_properties(
        data_file=data_file,
        days_threshold=days_threshold,
        dry_run=dry_run
    )
    
    logger.info(f"\nCleanup completed successfully!")
    return metrics

if __name__ == "__main__":
    main()
