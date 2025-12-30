"""
================================================================
DATA ENGINEER - MODULE 3: ETL/ELT & STREAMING
================================================================

ETL: Extract → Transform → Load (traditional)
ELT: Extract → Load → Transform (modern, với cloud warehouses)

Cài đặt: pip install apache-airflow pandas
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. ETL vs ELT:
   - ETL: Transform trước khi load, phù hợp on-premise
   - ELT: Load raw data, transform trong warehouse (Snowflake, BigQuery)

2. Orchestration Tools:
   - Apache Airflow: Most popular, DAG-based
   - Prefect: Modern alternative
   - Dagster: Data-aware orchestration

3. Data Quality:
   - Schema validation
   - Value checks (nulls, ranges)
   - Freshness checks
   - Tools: Great Expectations, deequ

4. Streaming vs Batch:
   - Batch: Xử lý theo intervals (hourly, daily)
   - Streaming: Real-time (Kafka, Kinesis, Flink)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import hashlib

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== SIMPLE ETL PIPELINE ==========

class SimpleETL:
    """ETL Pipeline đơn giản để học concepts"""
    
    def __init__(self, source_config: Dict, target_config: Dict):
        self.source = source_config
        self.target = target_config
        self.metrics = {
            'rows_extracted': 0,
            'rows_transformed': 0,
            'rows_loaded': 0,
            'errors': []
        }
    
    def extract(self) -> pd.DataFrame:
        """Extract data from source"""
        print(f"[EXTRACT] Reading from {self.source['type']}...")
        
        # Simulate reading from different sources
        if self.source['type'] == 'csv':
            # df = pd.read_csv(self.source['path'])
            df = pd.DataFrame({
                'id': range(1, 101),
                'name': [f'Product_{i}' for i in range(1, 101)],
                'price': [10.0 + i * 0.5 for i in range(100)],
                'category': ['A', 'B', 'C', 'D'] * 25,
                'created_at': [datetime.now() - timedelta(days=i) for i in range(100)]
            })
        elif self.source['type'] == 'api':
            # response = requests.get(self.source['url'])
            # df = pd.DataFrame(response.json())
            df = pd.DataFrame()  # Placeholder
        
        self.metrics['rows_extracted'] = len(df)
        print(f"[EXTRACT] Extracted {len(df)} rows")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        print("[TRANSFORM] Applying transformations...")
        
        # Clean data
        df = df.dropna()
        
        # Add derived columns
        df['price_category'] = pd.cut(
            df['price'], 
            bins=[0, 20, 40, 60, float('inf')],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # Add metadata
        df['_etl_timestamp'] = datetime.now()
        df['_source'] = self.source['type']
        df['_row_hash'] = df.apply(
            lambda row: hashlib.md5(str(row.values).encode()).hexdigest()[:16],
            axis=1
        )
        
        self.metrics['rows_transformed'] = len(df)
        print(f"[TRANSFORM] Transformed {len(df)} rows")
        return df
    
    def load(self, df: pd.DataFrame):
        """Load data to target"""
        print(f"[LOAD] Writing to {self.target['type']}...")
        
        if self.target['type'] == 'parquet':
            # df.to_parquet(self.target['path'], partition_cols=['category'])
            pass
        elif self.target['type'] == 'database':
            # df.to_sql(self.target['table'], engine, if_exists='append')
            pass
        
        self.metrics['rows_loaded'] = len(df)
        print(f"[LOAD] Loaded {len(df)} rows")
    
    def run(self):
        """Execute full ETL pipeline"""
        start_time = datetime.now()
        
        try:
            df = self.extract()
            df = self.transform(df)
            self.load(df)
            
            self.metrics['status'] = 'success'
            self.metrics['duration'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            self.metrics['status'] = 'failed'
            self.metrics['errors'].append(str(e))
            raise
        
        return self.metrics

# ========== AIRFLOW DAG (Pseudo-code) ==========

AIRFLOW_DAG = '''
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['alerts@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_sales_etl',
    default_args=default_args,
    description='Daily sales data pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
)

def extract_sales():
    """Extract from source database"""
    pass

def transform_sales():
    """Clean and transform data"""
    pass

def load_to_warehouse():
    """Load to data warehouse"""
    pass

def run_dbt_models():
    """Run dbt transformations"""
    pass

def validate_data():
    """Run data quality checks"""
    pass

# Define tasks
extract_task = PythonOperator(
    task_id='extract_sales',
    python_callable=extract_sales,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_sales',
    python_callable=transform_sales,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag,
)

dbt_task = BashOperator(
    task_id='run_dbt',
    bash_command='dbt run --models sales',
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Define dependencies
extract_task >> transform_task >> load_task >> dbt_task >> validate_task
'''

# ========== DATA QUALITY CHECKS ==========

class DataQualityChecker:
    """Simple data quality checker"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []
    
    def check_not_null(self, column: str):
        """Check column has no nulls"""
        null_count = self.df[column].isnull().sum()
        passed = null_count == 0
        self.results.append({
            'check': f'not_null:{column}',
            'passed': passed,
            'details': f'{null_count} null values found'
        })
        return passed
    
    def check_unique(self, column: str):
        """Check column values are unique"""
        duplicate_count = self.df[column].duplicated().sum()
        passed = duplicate_count == 0
        self.results.append({
            'check': f'unique:{column}',
            'passed': passed,
            'details': f'{duplicate_count} duplicates found'
        })
        return passed
    
    def check_range(self, column: str, min_val, max_val):
        """Check values are within range"""
        out_of_range = ((self.df[column] < min_val) | (self.df[column] > max_val)).sum()
        passed = out_of_range == 0
        self.results.append({
            'check': f'range:{column}[{min_val},{max_val}]',
            'passed': passed,
            'details': f'{out_of_range} values out of range'
        })
        return passed
    
    def check_freshness(self, date_column: str, max_age_hours: int):
        """Check data is fresh"""
        latest_date = self.df[date_column].max()
        age_hours = (datetime.now() - latest_date).total_seconds() / 3600
        passed = age_hours <= max_age_hours
        self.results.append({
            'check': f'freshness:{date_column}',
            'passed': passed,
            'details': f'Data is {age_hours:.1f} hours old'
        })
        return passed
    
    def summary(self):
        """Get summary of all checks"""
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        return {
            'total_checks': len(self.results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.results) if self.results else 0,
            'details': self.results
        }

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Extend SimpleETL:
       - Thêm incremental load (chỉ load records mới/changed)
       - Thêm checkpoint để resume khi failed
       - Thêm logging chi tiết

BÀI 2: Setup Airflow:
       - pip install apache-airflow
       - Tạo DAG cho pipeline của bạn
       - Thêm sensors để wait for data

BÀI 3: Implement Change Data Capture (CDC):
       - Track INSERT, UPDATE, DELETE
       - Apply changes to target incrementally

BÀI 4: Streaming pipeline với Kafka:
       - Producer: Generate events
       - Consumer: Process real-time
       - Sink: Write to database
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== ETL Pipeline Demo ===\n")
    
    # Run ETL
    etl = SimpleETL(
        source_config={'type': 'csv', 'path': 'data.csv'},
        target_config={'type': 'parquet', 'path': 'output/'}
    )
    metrics = etl.run()
    print(f"\n[COMPLETE] Metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # Data Quality
    print("\n=== Data Quality Checks ===")
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 150, 40, 50],  # 150 is out of range
        'date': [datetime.now() - timedelta(hours=i) for i in range(5)]
    })
    
    checker = DataQualityChecker(df)
    checker.check_not_null('id')
    checker.check_unique('id')
    checker.check_range('value', 0, 100)
    checker.check_freshness('date', 24)
    
    print(json.dumps(checker.summary(), indent=2, default=str))
