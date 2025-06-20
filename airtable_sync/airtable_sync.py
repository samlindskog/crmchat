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
import httpx
import asyncio

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
        AIRTABLE_TEST_MODE    Enable test mode (default: false)
        XAI_API_KEY           Your xAI API key
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

class AirtableFetcher:
    def __init__(self):
        # Required environment variables
        self.api_key = os.getenv('AIRTABLE_API_KEY')
        self.base_id = os.getenv('AIRTABLE_BASE_ID')
        self.output_dir = Path(os.getenv('AIRTABLE_OUTPUT_DIR', 'csv_output'))
        self.sync_interval = int(os.getenv('AIRTABLE_SYNC_INTERVAL', '60'))  # minutes
        self.tables = os.getenv('AIRTABLE_TABLES', 'Table1,Table2').split(',')
        self.rate_limit = float(os.getenv('AIRTABLE_RATE_LIMIT', '5'))  # requests per second
        self.error_wait = int(os.getenv('AIRTABLE_ERROR_WAIT', '35'))  # seconds
        self.xai_api_key = os.getenv('XAI_API_KEY')
        self.test_mode = os.getenv('AIRTABLE_TEST_MODE', 'false').lower() in ['1', 'true']

        if not self.api_key or not self.base_id:
            raise ValueError("AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set in environment")

        self.api = Api(self.api_key)
        self.base = self.api.base(self.base_id)
        self.rate_limiter = RateLimiter(self.rate_limit)

    def _make_api_call(self, table_name: str) -> List[List[Dict]]:
        table = self.base.table(table_name)
        records = []
        total_records = 0
        for page in table.iterate():
            self.rate_limiter.wait()
            if self.test_mode:
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
                records.append(page)
        return records

    def get_table_data(self, table_name: str) -> List[List[Dict]]:
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

class AirtableExporter:
    def __init__(self, fetcher: AirtableFetcher):
        self.fetcher = fetcher
        self.output_dir = Path(os.getenv('AIRTABLE_OUTPUT_DIR', 'csv_output'))
        self.xai_api_key = os.getenv('XAI_API_KEY')

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        cleaned = re.sub(r'[^a-zA-Z0-9,!?;:\.\'\"\(\)\s@<>_#=+%&~$*\-\/\[\]]', '', text)
        return cleaned

    def save_to_csv(self, table_name: str, records_pages: List[List[Dict]]):
        if not records_pages:
            logger.warning(f"No records to save for table {table_name}")
            return
        all_records = []
        for page in records_pages:
            for record in page:
                fields = {k: self.clean_text(v) if isinstance(v, (str,list)) else '' for k, v in record['fields'].items()}
                fields['airtable_record_id'] = record['id']
                fields['airtable_table'] = table_name
                all_records.append(fields)
        if all_records:
            columns = sorted(set().union(*(d.keys() for d in all_records)))
            priority_columns = ['airtable_record_id', 'airtable_table']
            other_columns = [col for col in columns if col not in priority_columns]
            columns = priority_columns + other_columns
            df = pd.DataFrame(all_records, columns=columns)
        else:
            df = pd.DataFrame()
        filename = self.output_dir / f"{table_name}.csv"
        df.to_csv(
            filename,
            index=False,
            quoting=1,
            escapechar='\\',
            doublequote=True,
            lineterminator='\n',
            na_rep=''
        )
        logger.info(f"Updated {len(all_records)} records in {filename}")

    def linkedin_xai_summary(self, linkedin_url_list: str, first_name: str = '', last_name: str = '') -> str:
        xai_api_key = self.xai_api_key
        if not xai_api_key or not linkedin_url_list:
            return 'Summary not available.'
        try:
            api_url = 'https://api.x.ai/v1/chat/completions'
            headers = {
                'Authorization': f'Bearer {xai_api_key}',
                'Content-Type': 'application/json',
            }
            name_part = f" for {first_name} {last_name}".strip() if first_name or last_name else ''
            data = {
                'model': 'grok-3',
                'messages': [
                    {'role': 'system', 'content': f'''You are a helpful assistant that provides context-aware responses
                     about a person by drawing on linkedin profileinformation from the list of URLs that follow:
                     {linkedin_url_list}.'''},
                    {'role': 'user', 'content': f'''summarize professional history, experience, knowledge, education,
                     and other relevant information {name_part}. You must be specific and detailed. Organize relevant
                     information into sections.'''}
                ],
            }
            resp = requests.post(api_url, headers=headers, json=data, timeout=90)
            if resp.status_code == 200:
                result = resp.json()
                return result.get('choices', [{}])[0].get('message', {}).get('content', 'Summary not available.')
            else:
                logger.warning(f"xAI API error: {resp.status_code} {resp.text}")
                return 'Summary not available.'
        except Exception as e:
            logger.warning(f"Exception calling xAI API: {e}")
            return 'Summary not available.'

    def save_to_plaintext(self, table_name: str, records_pages: List[List[Dict]]):
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
            linkedin_url = record.get('LinkedIn Profile', '')
            if isinstance(linkedin_url, list):
                linkedin_url = linkedin_url[0] if linkedin_url else ''
            linkedin_url = str(linkedin_url).strip()
            first_name = str(record.get('First Name', '')).strip()
            last_name = str(record.get('Last Name', '')).strip()
            if linkedin_url.lower().startswith('http'):
                paths = [
                    'details/experience/', 'details/certifications/', 'details/education/',
                    'details/recommendations/', 'details/projects/', 'details/skills/',
                    'details/courses/', 'details/languages/'
                ]
                url_list = "\n".join([f"{linkedin_url.rstrip('/')}/{path}" for path in paths])
                summary = self.linkedin_xai_summary(url_list, first_name, last_name)
            else:
                summary = 'No LinkedIn URL provided.'
            plaintext_content += f"### LinkedIn Summary ###\n{summary}\n"
            if idx < len(all_records) - 1:
                plaintext_content += "---\n"
        filename = self.output_dir / f"{table_name}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(plaintext_content)
        logger.info(f"Saved {len(all_records)} records to plaintext file {filename}")

    def sync_table(self, table_name: str):
        logger.info(f"Syncing table: {table_name}")
        records = self.fetcher.get_table_data(table_name)
        self.save_to_csv(table_name, records)
        self.save_to_plaintext(table_name, records)

    def sync_all_tables(self):
        start_time = time.time()
        for table_name in self.fetcher.tables:
            self.sync_table(table_name)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Sync completed in {duration:.2f} seconds")

def main():
    try:
        fetcher = AirtableFetcher()
        exporter = AirtableExporter(fetcher)

        # Log environment variables
        logger.info(f"Output directory set to: {fetcher.output_dir.absolute()}")
        logger.info(f"Sync interval set to: {fetcher.sync_interval} minutes")
        logger.info(f"Tables to sync: {', '.join(fetcher.tables)}")
        logger.info(f"Rate limit set to: {fetcher.rate_limit} requests per second")
        logger.info(f"Error wait time set to: {fetcher.error_wait} seconds")
        logger.info(f"Test mode: {'ENABLED' if fetcher.test_mode else 'disabled'}")
        if not fetcher.xai_api_key:
            logger.warning('XAI_API_KEY not set. LinkedIn summaries will be skipped.')
        
        exporter.sync_all_tables()
        schedule.every(fetcher.sync_interval).minutes.do(exporter.sync_all_tables)
        logger.info(f"Starting sync service with {fetcher.sync_interval} minute interval...")
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    show_help()  # Show help if -h or --help is used
    main()