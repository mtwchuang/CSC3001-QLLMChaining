# libraries
import sqlite3
import hashlib
import csv
import os
import pandas as pd
from faker import Faker
import random

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML

###################################################
# Invoice DB Function Calls
###################################################

def connect_to_db():
    """Creates sqlite invoice database if not exists, else returns connection"""
    conn = sqlite3.connect('database/invoice.db')
    return conn

def drop_db():
    """Drops sqlite3 invoice database"""
    try:
        os.remove("database/invoice.db")
        print(f"Database dropped successfully.")
    except FileNotFoundError:
        print(f"Database does not exist.")
    except Exception as e:
        print(f"Error dropping database: {e}")

def create_table(c, table_name, columns, replace_flag):
    """Creates a new table using defined table name and column structure."""
    try:
        # Removes table if replace flag is true
        if replace_flag:
            try:
                c.execute(f"DROP TABLE IF EXISTS {table_name}")
            except sqlite3.Error as e:
                print(f"Error dropping table {table_name}: {e}")
                return False
        # Parses columns and datatypes into string, and stores in column_def
        columns_def = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        # Create data table
        try:
            c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_def})")
            return True
        except sqlite3.Error as e:
            print(f"Error creating table {table_name}: {e}")
            return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    
def get_db_schema():
    """Retrieve the schema (DDL) for a table from an SQLite database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    try:
        # SQL command that fetches context from Database
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schema_tuples = cursor.fetchall()
        # Filter out hash tables from being included in the schema retrieved
        schema_tuples = [x for x in schema_tuples if not x[0].startswith('CREATE TABLE ') or not x[0].split(' ')[2].endswith('_HASH')]
        # Extract the schemas from the tuples and join them into a single string
        schema = '\n'.join(x[0] for x in schema_tuples)
        print("Retrieving Database Schema")
        return schema
    except sqlite3.Error as e:
        print(f"Error retrieving schema: {e}")
    finally:
        if conn:
            conn.close()

def populate_table(c, table_name, csv_file, id_column):
    """Populates the target data table and its hash table with data from a CSV file."""
    try:
        with open(csv_file, 'r') as fin:
            try:
                dr = csv.DictReader(fin)
                data = [tuple(i[field] for field in dr.fieldnames) for i in dr]
            except csv.Error as e:
                print(f"Error reading CSV file {csv_file}: {e}")
                return False
        
        insert_data_with_hash(c, table_name, data, id_column)
        print(f"Data inserted into table {table_name} and {table_name}_HASH successfully.")
        return True
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def create_invoice_db():
    """Setting up the Invoice Database and Tables."""
    try:
        # Connection and Cursors
        drop_db()
        conn = connect_to_db()
        c = conn.cursor()

        # Table definition
        tables = {
            'FT_INVOICE': {
                'INVOICE_ID': 'INTEGER PRIMARY KEY',
                'DEPT_ID': 'INTEGER',
                'PRODUCT_ID': 'INTEGER',
                'PAYMENT_AMOUNT': 'INTEGER',
                'VENDOR_ID': 'INTEGER',
                'EMPLOYEE_ID': 'INTEGER'
            },
            'LU_EMPLOYEE': {
                'EMPLOYEE_ID': 'INTEGER PRIMARY KEY',
                'EMPLOYEE_NAME': 'TEXT'
            },
            'LU_PRODUCT': {
                'PRODUCT_ID': 'INTEGER PRIMARY KEY',
                'PRODUCT_NAME': 'TEXT'
            },
            'LU_VENDOR': {
                'VENDOR_ID': 'INTEGER PRIMARY KEY',
                'VENDOR_FULLNAME': 'TEXT',
                'COUNTRY_CODE': 'TEXT'
            },
            'LU_DEPARTMENT': {
                'DEPT_ID': 'INTEGER PRIMARY KEY',
                'DEPT_NAME': 'TEXT'
            }
        }

        # Iterate through the dictionary and creates a original table and its corresponding hashed table
        for table_name, columns in tables.items():
            create_table(c, table_name, columns, replace_flag=False)
            # Retrieves id values from the identifiying column for table and stores it in id_column
            id_column = next(col for col, dtype in columns.items() if 'PRIMARY KEY' in dtype)
            create_hash_table(c, table_name, id_column, replace_flag=False)

        # Define data sources
        csv_files = {
            'FT_INVOICE': 'raw_datasets/ft_invoice.csv',
            'LU_EMPLOYEE': 'raw_datasets/lu_employee.csv',
            'LU_PRODUCT': 'raw_datasets/lu_product.csv',
            'LU_VENDOR': 'raw_datasets/lu_vendor.csv',
            'LU_DEPARTMENT': 'raw_datasets/lu_department.csv'
        }

        # Iterate for each table and populate table
        for table_name, csv_file in csv_files.items():
            id_column = next(col for col, dtype in tables[table_name].items() if 'PRIMARY KEY' in dtype)
            populate_table(c, table_name, csv_file, id_column)
        
        print("Database created successfully")
        conn.commit()
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

def generate_invoice_data():
    """Generation of random synthetic data mocking up data in Invoice Database"""
    # Initialize Faker
    fake = Faker()
    Faker.seed(1903)
    random.seed(1903)

    # Define the number of records
    num_unique_employees = 20
    num_unique_vendors = 25
    num_unique_products = 20
    num_unique_departments = 5
    num_records = 1000

    # Mock data
    vendor_names = ["Vendor A", "Vendor B", "Vendor C", "Vendor D", "Vendor E", "Vendor F", "Vendor G", "Vendor H", "Vendor I", "Vendor J", "Vendor K", "Vendor L", "Vendor M", "Vendor N", "Vendor O", "Vendor P", "Vendor Q", "Vendor R", "Vendor S", "Vendor T", "Vendor U", "Vendor V", "Vendor W", "Vendor X", "Vendor Y"]
    product_names = ["Product 1", "Product 2", "Product 3", "Product 4", "Product 5", "Product 6", "Product 7", "Product 8", "Product 9", "Product 10", "Product 11", "Product 12", "Product 13", "Product 14", "Product 15", "Product 16", "Product 17", "Product 18", "Product 19", "Product 20"]
    country_codes = ["USA", "CAN", "MEX", "BRA", "ARG", "GBR", "FRA", "DEU", "ITA", "ESP", "CHN", "JPN", "KOR", "IND", "AUS", "NZL", "ZAF", "NGA", "EGY", "RUS", "SAU", "IRN", "TUR", "IDN", "SGP"]
    department_names = ["Sales", "Marketing", "Finance", "Human Resources", "IT"]

    # Generate synthetic data for LU_EMPLOYEE
    lu_employee = pd.DataFrame({
        'EMPLOYEE_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_employees)],
        'EMPLOYEE_NAME': [fake.name() for _ in range(num_unique_employees)]
    })

    # Generate synthetic data for LU_VENDOR
    lu_vendor = pd.DataFrame({
        'VENDOR_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_vendors)],
        'VENDOR_FULLNAME': vendor_names[:num_unique_vendors],
        'COUNTRY_CODE': country_codes[:num_unique_vendors]
    })

    # Generate synthetic data for LU_PRODUCT
    lu_product = pd.DataFrame({
        'PRODUCT_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_products)],
        'PRODUCT_NAME': product_names[:num_unique_products],
    })

    # Generate synthentic data for LU_DEPARTMENT
    lu_department = pd.DataFrame({
        'DEPT_ID': [fake.unique.random_number(digits=3) for _ in range(num_unique_departments)],
        'DEPT_NAME': department_names[:num_unique_departments],

    })

    # Generate synthetic data for Fact Table
    fact_table = pd.DataFrame({
        'INVOICE_ID': [fake.unique.random_number(digits=5) for _ in range(num_records)],
        'DEPT_ID': [random.choice(lu_department['DEPT_ID']) for _ in range(num_records)],
        'PRODUCT_ID': [random.choice(lu_product['PRODUCT_ID']) for _ in range(num_records)],
        'PAYMENT_AMOUNT': [random.randint(10,100000) for _ in range(num_records)],
        'VENDOR_ID': [random.choice(lu_vendor['VENDOR_ID']) for _ in range(num_records)],
        'EMPLOYEE_ID': [random.choice(lu_employee['EMPLOYEE_ID']) for _ in range(num_records)],
    })

    # Export to CSV
    fact_table.to_csv('raw_datasets/ft_invoice.csv', index=False)
    lu_employee.to_csv('raw_datasets/lu_employee.csv', index=False)
    lu_vendor.to_csv('raw_datasets/lu_vendor.csv', index=False)
    lu_product.to_csv('raw_datasets/lu_product.csv', index=False)
    lu_department.to_csv('raw_datasets/lu_department.csv', index=False)

def extract_table_names(parsed_query):
    """Get names of tables that were involved in a Query"""
    tables = set()
    for statement in parsed_query:
        if statement.get_type() == 'SELECT':
            from_seen = False
            join_seen = False
            for token in statement.tokens:
                if from_seen and isinstance(token, Identifier):
                    tables.add(token.get_real_name())
                if join_seen and isinstance(token, Identifier):
                    tables.add(token.get_real_name())
                if token.ttype is Keyword and token.value.upper() == 'FROM':
                    from_seen = True
                if token.ttype is Keyword and token.value.upper() == 'JOIN':
                    join_seen = True
                if token.ttype is Keyword and token.value.upper() in ['WHERE', 'GROUP BY', 'ORDER BY']:
                    from_seen = False
                    join_seen = False
    return list(tables)

def extract_primary_keys(c, table_names):
    primary_keys = {}
    for table in table_names:
        c.execute(f"PRAGMA table_info({table})")
        for col in c.fetchall():
            if col[5] == 1:  # Primary key column
                primary_keys[table] = col[1]
                break  # Assuming there is only one primary key column per table
    # print(primary_keys)
    return primary_keys

def print_first_few_rows(data, num_rows=5, label="Data"):
    print(f"--- {label} (first {num_rows} rows) ---")
    for row in data[:num_rows]:
        print(row)
    print(f"--- End of {label} ---")

###################################################
# Hashing DB Function Calls
###################################################

def create_hash_table(c, table_name, id_column, replace_flag):
    """Creates a hash table for a given data table."""
    # Appends "_HASH" to the name or original table
    hash_table_name = f"{table_name}_HASH"
    # Column definition
    columns = {
        id_column: 'INTEGER PRIMARY KEY',
        'ROW_HASHED_VALUE': 'TEXT'
    }
    return create_table(c, hash_table_name, columns, replace_flag)

def generate_hash(row):
    """Generates SHA256 hash for a given row."""
    data = ''.join(str(value) for value in row)
    return hashlib.sha256(data.encode()).hexdigest()

def insert_data_with_hash(c, table_name, data, id_column):
    """Inserts data into the original table and its hash table."""
    hash_table_name = f"{table_name}_HASH"
    # Get the column names of the table
    fieldnames = [desc[1] for desc in c.execute(f"PRAGMA table_info({table_name})").fetchall()]
    # Inserts data into original data, using placeholders with "?" values to form query, which will be replaced with actual data in the executemany() command
    placeholders = ', '.join('?' * len(fieldnames))
    query = f"INSERT INTO {table_name} ({', '.join(fieldnames)}) VALUES ({placeholders})"
    c.executemany(query, data)
    
    # Uses index() function to extract the id values from the id column
    id_index = fieldnames.index(id_column)
    # Prepare data for the hash table
    data_with_hash = [(row[id_index], generate_hash(row)) for row in data]
    
    # Get the column names of the hash table
    hash_fieldnames = [id_column, 'ROW_HASHED_VALUE']
    # Inserts data into hashed data, using placeholders with "?" values to form query, which will be replaced with actual data in the executemany() command
    placeholders = ', '.join('?' * len(hash_fieldnames))
    query = f"INSERT INTO {hash_table_name} ({', '.join(hash_fieldnames)}) VALUES ({placeholders})"
    c.executemany(query, data_with_hash)

def fetch_and_verify_table(c, table_name, primary_key):
    compromised_rows = []
    total_rows = 0
    verified_rows = 0

    # Fetch the entire table
    c.execute(f"SELECT * FROM {table_name}")
    table_data = c.fetchall()
    total_rows = len(table_data)

    for row in table_data:
        id_value = row[0]  # Assuming the first column is the primary key
        data = row
        
        # Fetch the stored hash from the hash table
        c.execute(f"SELECT ROW_HASHED_VALUE FROM {table_name}_HASH WHERE {primary_key} = ?", (id_value,))
        stored_hash = c.fetchone()

        if stored_hash:
            # Compute the hash of the full row
            computed_hash = generate_hash(data)
            if computed_hash == stored_hash[0]:
                verified_rows += 1
            else:
                compromised_rows.append(row)
        else:
            compromised_rows.append(row)

    return compromised_rows, total_rows, verified_rows

def verify_data_integrity(c, query):
    parsed = sqlparse.parse(query)
    table_names = extract_table_names(parsed)
    primary_keys = extract_primary_keys(c, table_names)

    compromised_data = {}
    table_summaries = {}

    # Fetch complete rows and verify integrity for each table
    for table in table_names:
        primary_key = primary_keys[table]
        c_data, total_rows, verified_rows = fetch_and_verify_table(c, table, primary_key)
        compromised_data[table] = c_data
        table_summaries[table] = {
            "total_rows": total_rows,
            "verified_rows": verified_rows,
            "percentage_verified": (verified_rows / total_rows * 100) if total_rows > 0 else 0
        }

    return compromised_data, table_summaries

def get_table_summaries(summaries):
    """Prepare table summaries for display."""
    summary_data = []
    for table, summary in summaries.items():
        summary_data.append({
            "Table": table,
            "Total Rows": summary['total_rows'],
            "Verified Rows": summary['verified_rows'],
            "Percentage Verified": f"{summary['percentage_verified']:.2f}%"
        })
    return summary_data

###################################################
# Benchmarking DB Function Calls
###################################################

def connect_to_benchmark_db():
    """Creates sqlite benchmarking database if not exists, else returns connection"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
    # Create table to store SQL benchmark statistics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sql_benchmark_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            correct_result BOOLEAN,
            bleu_score REAL,
            difficulty TEXT
        )
    ''')
    print("Initialized sql_benchmark_stats table")
    # Create table to store summarized SQL benchmark statistics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sql_summary_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            difficulty TEXT, 
            accuracy REAL, 
            average_bleu_score REAL,
            average_latency REAL
        )
    ''')
    print("Initialized sql_summary_stats table")

    # Create table to store CHAT benchmark statistics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_benchmark_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            bert_score REAL,
            rouge_score REAL,
            difficulty TEXT
        )
    ''')
    print("Initialized chat_benchmark_stats table")
    
    # Create table to store summarized CHAT benchmark statistics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_summary_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            difficulty TEXT, 
            average_bert_score REAL, 
            average_rouge_score REAL,
            average_latency REAL
        )
    ''')
    print("Initialized chat_summary_stats table")

    # Create table to store general visualization benchmark statistics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vis_benchmark_stats (
            query TEXT,
            executable BOOLEAN,
            ssim_index REAL,
            pixel_similarity REAL,
            bleu_score REAL,
            difficulty TEXT
        )
    ''')
    print("Initialized vis_benchmark_stats table")
    # Create table to store summarized visualized benchmark statistics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vis_summary_stats (
            difficulty TEXT PRIMARY KEY,
            executable_rate REAL,
            average_ssim_index REAL,
            average_pixel_similarity REAL,
            average_bleu_score REAL,
            average_latency REAL
        )
    ''')
    print("Initialized vis_summary_stats table")
    conn.commit()
    conn.close()
    return conn

def drop_benchmark_db():
    """Drops sqlite3 benchmarking database"""
    try:
        os.remove("database/benchmarking.db")
        print(f"Database dropped successfully.")
    except FileNotFoundError:
        print(f"Database does not exist.")
    except Exception as e:
        print(f"Error dropping database: {e}")

 ####### TEXT TO SQL #######
def store_sql_summary(difficulty, accuracy, avg_bleu, avg_latency):
    """Insert SQL summary for a particular difficulty into sql_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
        
    cursor.execute('''
        INSERT OR REPLACE INTO sql_summary_stats (difficulty, accuracy, average_bleu_score, average_latency)
        VALUES (?, ?, ?, ?)
    ''', (difficulty, accuracy, avg_bleu, avg_latency))
    conn.commit()
    conn.close()

def store_sql_stat(query, correct_result, bleu_score, difficulty):
    """Insert SQL statistic for a particular query into sql_benchmark_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sql_benchmark_stats (query, correct_result, bleu_score, difficulty)
        VALUES (?, ?, ?, ?)
    ''', (query, correct_result, bleu_score, difficulty))
    conn.commit()
    conn.close()

def read_sql_summary():
    """Read the SQL summary statistics from sql_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    df = pd.read_sql_query("SELECT * FROM sql_summary_stats", conn)
    conn.close()
    return df

 ####### TEXT TO CHAT #######
def store_chat_summary(difficulty, avg_bert, avg_rouge, avg_latency):
    """Insert chat summary for a particular difficulty into chat_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
        
    cursor.execute('''
        INSERT OR REPLACE INTO chat_summary_stats (difficulty, average_bert_score, average_rouge_score, average_latency)
        VALUES (?, ?, ?, ?)
    ''', (difficulty, float(avg_bert), avg_rouge, avg_latency))
    conn.commit()
    conn.close()

def store_chat_stat(query, bert_score, rouge_score, difficulty):
    """Insert chat statistic for a particular query into chat_benchmark_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_benchmark_stats (query, bert_score, rouge_score, difficulty)
        VALUES (?, ?, ?, ?)
    ''', (query, float(bert_score), rouge_score, difficulty))
    conn.commit()
    conn.close()

def read_chat_summary():
    """Read the chat summary statistics from chat_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    df = pd.read_sql_query("SELECT * FROM chat_summary_stats", conn)
    conn.close()
    return df

 ####### TEXT TO VIS #######
def store_vis_stat(query, executable, ssim_index, pixel_similarity, bleu_score, difficulty):
    """Insert visualization statistic for a particular query into vis_benchmark_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO vis_benchmark_stats (query, executable, ssim_index, pixel_similarity, bleu_score, difficulty)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (query, executable, ssim_index, pixel_similarity, bleu_score, difficulty))
    conn.commit()
    conn.close()

def store_vis_summary(difficulty, exec_rate, avg_ssim, avg_pixel_similarity, avg_bleu, avg_latency):
    """Insert visualization summary for a particular difficulty into vis_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    cursor = conn.cursor()
        
    cursor.execute('''
        INSERT OR REPLACE INTO vis_summary_stats (difficulty, executable_rate, average_ssim_index, average_pixel_similarity, average_bleu_score, average_latency)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (difficulty, exec_rate, avg_ssim, avg_pixel_similarity, avg_bleu, avg_latency))
    
    conn.commit()
    conn.close()

def read_vis_summary():
    """Read the visualization summary statistics from vis_summary_stats table"""
    conn = sqlite3.connect('database/benchmarking.db')
    df = pd.read_sql_query("SELECT * FROM vis_summary_stats", conn)
    conn.close()
    return df