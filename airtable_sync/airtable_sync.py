import os
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from ratelimit import limits, sleep_and_retry
from pyairtable.exceptions import PyAirtableError
import re
import idna
import requests

import pandas as pd
from pyairtable import Api
import schedule

def show_help():
    parser = argparse.ArgumentParser(
        description='Sync Airtable data to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Environment Variables:
    Required:
        AIRTABLE_API_KEY      Your Airtable API key
        AIRTABLE_BASE_ID      Your Airtable base ID

    Optional:
        AIRTABLE_OUTPUT_DIR   Directory to save CSV files (default: csv_output)
        AIRTABLE_SYNC_INTERVAL Sync interval in minutes (default: 60)
        AIRTABLE_TABLES       Comma-separated list of tables to sync (default: Table1,Table2)
        AIRTABLE_SYNC_LOG_LEVEL Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)
        AIRTABLE_RATE_LIMIT   Maximum requests per second (default: 5)
        AIRTABLE_ERROR_WAIT   Seconds to wait after API error (default: 35)
        PERPLEXITY_API_KEY    Your Perplexity AI API key

Example:
    AIRTABLE_API_KEY=key123 AIRTABLE_BASE_ID=base456 AIRTABLE_SYNC_LOG_LEVEL=DEBUG python airtable_sync.py
''')
    parser.parse_args()

# Set up logging with configurable level
log_level = os.getenv('AIRTABLE_SYNC_LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')

logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    def wait(self):
        """Wait if necessary to respect the rate limit."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            logger.debug(f"Rate limiting: waiting {sleep_time:.3f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

class AirtableSync:
    def __init__(self):
        # Required environment variables
        self.api_key = os.getenv('AIRTABLE_API_KEY')
        self.base_id = os.getenv('AIRTABLE_BASE_ID')
        
        # Optional environment variables with defaults
        self.output_dir = Path(os.getenv('AIRTABLE_OUTPUT_DIR', 'csv_output'))
        self.sync_interval = int(os.getenv('AIRTABLE_SYNC_INTERVAL', '60'))  # minutes
        self.tables = os.getenv('AIRTABLE_TABLES', 'Table1,Table2').split(',')
        self.rate_limit = float(os.getenv('AIRTABLE_RATE_LIMIT', '5'))  # requests per second
        self.error_wait = int(os.getenv('AIRTABLE_ERROR_WAIT', '35'))  # seconds
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.test_mode = os.getenv('AIRTABLE_TEST_MODE', '').lower() in ['1', 'true']
        
        if not self.api_key or not self.base_id:
            raise ValueError("AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set in environment")
        
        self.api = Api(self.api_key)
        self.base = self.api.base(self.base_id)
        self.rate_limiter = RateLimiter(self.rate_limit)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        logger.info(f"Sync interval set to: {self.sync_interval} minutes")
        logger.info(f"Tables to sync: {', '.join(self.tables)}")
        logger.info(f"Rate limit set to: {self.rate_limit} requests per second")
        logger.info(f"Error wait time set to: {self.error_wait} seconds")
        logger.debug(f"Log level set to: {log_level}")
        logger.info(f"Test mode: {'ENABLED' if self.test_mode else 'disabled'}")
        
        if not self.perplexity_api_key:
            logger.warning('PERPLEXITY_API_KEY not set. LinkedIn summaries will be skipped.')
    
    def _make_api_call(self, table_name: str) -> List[List[Dict]]:
        """Make an API call to Airtable with rate limiting."""
        table = self.base.table(table_name)
        records = []  # List of pages, where each page is a list of records
        total_records = 0
        # Use iterate() to fetch records one page at a time
        for page in table.iterate():
            self.rate_limiter.wait()  # Rate limit each page fetch
            if self.test_mode:
                # Only collect up to 3 records in test mode
                remaining = 3 - total_records
                if remaining <= 0:
                    break
                if len(page) > remaining:
                    records.append(page[:remaining])
                    total_records += remaining
                    break
                else:
                    records.append(page)
                    total_records += len(page)
            else:
                records.append(page)  # Store each page as a separate element
        return records
    
    def get_table_data(self, table_name: str) -> List[List[Dict]]:
        """Fetch all records from a specific Airtable table with rate limiting."""
        try:
            records = self._make_api_call(table_name)
            logger.debug(f"Fetched records from table {table_name}")
            return records
        except PyAirtableError as e:
            logger.warning(f"Airtable API error for table {table_name}: {str(e)}")
            logger.warning(f"Waiting {self.error_wait} seconds before continuing...")
            time.sleep(self.error_wait)
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching data from table {table_name}: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Remove unwanted symbols from text."""
        if not isinstance(text, str):
            return str(text)
        
        # Then keep only allowed characters
        cleaned = re.sub(r'[^a-zA-Z0-9,!?;:\.\'\"\(\)\s@<>_#=+%&~$*\-\/\[\]]', '', text)
        
        return cleaned

    def save_to_csv(self, table_name: str, records_pages: List[List[Dict]]):
        """Save records to a CSV file, overwriting if it exists."""
        if not records_pages:
            logger.warning(f"No records to save for table {table_name}")
            return
        
        # Create a list to hold all records
        all_records = []
        for page in records_pages:
            for record in page:
                # Ensure all values are strings and clean non-English characters
                fields = {k: self.clean_text(v) if isinstance(v, (str,list)) else '' for k, v in record['fields'].items()}
                # Add Airtable record ID and table name
                fields['airtable_record_id'] = record['id']
                fields['airtable_table'] = table_name
                all_records.append(fields)
        
        # Convert to DataFrame with explicit column order
        if all_records:
            # Get all unique keys from all records
            columns = sorted(set().union(*(d.keys() for d in all_records)))
            # Ensure airtable_record_id and airtable_table are first columns
            priority_columns = ['airtable_record_id', 'airtable_table']
            other_columns = [col for col in columns if col not in priority_columns]
            columns = priority_columns + other_columns
            df = pd.DataFrame(all_records, columns=columns)
        else:
            df = pd.DataFrame()
        
        # Use a fixed filename for each table
        filename = self.output_dir / f"{table_name}.csv"
        
        # Save to CSV with strict formatting options
        df.to_csv(
            filename,
            index=False,
            quoting=1,  # Quote all non-numeric fields
            escapechar='\\',  # Use backslash as escape character
            doublequote=True,  # Double up quotes to escape them
            lineterminator='\n',  # Use Unix-style line endings
            na_rep=''  # Replace NaN with empty string
        )
        logger.info(f"Updated {len(all_records)} records in {filename}")

    def summarize_with_perplexity(self, linkedin_url: str, first_name: str = '', last_name: str = '') -> str:
        """Send the LinkedIn URL and name to Perplexity AI for summarization."""
        if not self.perplexity_api_key or not linkedin_url:
            return 'Summary not available.'
        try:
            api_url = 'https://api.perplexity.ai/chat/completions'
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json',
            }
            name_part = f" for {first_name} {last_name}".strip() if first_name or last_name else ''
            prompt = f'summarize professional history in up to 250 words{name_part}. {linkedin_url}. strictly provide only the final answer, I do not want any explanation or steps or context'
            data = {
                'model': 'sonar-reasoning-pro',
                'messages': [
                    {'role': 'system', 'content': 'strictly provide only the final answer, I do not want any explanation or steps or context'},
                    {'role': 'user', 'content': prompt}
                ],
                'search_domain_filter': [
                    f'{linkedin_url}'
                ]
            }
            resp = requests.post(api_url, headers=headers, json=data, timeout=90)
            if resp.status_code == 200:
                result = resp.json()
                return result.get('choices', [{}])[0].get('message', {}).get('content', 'Summary not available.')
            else:
                logger.warning(f"Perplexity API error: {resp.status_code} {resp.text}")
                return 'Summary not available.'
        except Exception as e:
            logger.warning(f"Exception calling Perplexity API: {e}")
            return 'Summary not available.'

    def save_to_plaintext(self, table_name: str, records_pages: List[List[Dict]]):
        """Save records from a specific table to a plaintext file."""
        if table_name != 'tblgitdXggPeL7cqD':
            return
            
        if not records_pages:
            logger.warning(f"No records to save to plaintext for table {table_name}")
            return

        all_records = []
        for page in records_pages:
            for record in page:
                fields = {k: self.clean_text(v) if isinstance(v, (str,list)) else '' for k, v in record['fields'].items()}
                all_records.append(fields)

        plaintext_content = ""
        for idx, record in enumerate(all_records):
            for column, value in record.items():
                plaintext_content += f"### {column} ###\n{value}\n"
            # LinkedIn summary section
            linkedin_url = record.get('LinkedIn Profile', '')
            if isinstance(linkedin_url, list):
                linkedin_url = linkedin_url[0] if linkedin_url else ''
            linkedin_url = str(linkedin_url).strip()
            first_name = str(record.get('First Name', '')).strip()
            last_name = str(record.get('Last Name', '')).strip()
            if linkedin_url.lower().startswith('http'):
                summary = self.summarize_with_perplexity(linkedin_url, first_name, last_name)
            else:
                summary = 'No LinkedIn URL provided.'
            plaintext_content += f"### LinkedIn Summary ###\n{summary}\n"
            # Add separator if not the last record
            if idx < len(all_records) - 1:
                plaintext_content += "---\n"

        filename = self.output_dir / f"{table_name}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(plaintext_content)
        
        logger.info(f"Saved {len(all_records)} records to plaintext file {filename}")

    def sync_table(self, table_name: str):
        """Sync a single table from Airtable to CSV."""
        logger.info(f"Syncing table: {table_name}")
        records = self.get_table_data(table_name)
        self.save_to_csv(table_name, records)
        self.save_to_plaintext(table_name, records)
    
    def sync_all_tables(self):
        """Sync all configured tables from Airtable to CSV."""
        start_time = time.time()
        for table_name in self.tables:
            self.sync_table(table_name)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Sync completed in {duration:.2f} seconds")

def main():
    try:
        sync = AirtableSync()
        
        # Run initial sync
        sync.sync_all_tables()
        
        # Schedule sync based on interval
        schedule.every(sync.sync_interval).minutes.do(sync.sync_all_tables)
        
        logger.info(f"Starting sync service with {sync.sync_interval} minute interval...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending tasks
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    show_help()  # Show help if -h or --help is used
    main()