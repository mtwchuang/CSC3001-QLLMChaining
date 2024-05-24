# libraries
import sqlite3
import csv
import os
import pandas as pd

# Sqlite Function Calls
def connect_to_db():
    """Creates sqlite database if not exists, else returns connection"""
    conn = sqlite3.connect('invoice_database.db')
    return conn

def drop_db():
    """Drop the SQLite database."""
    try:
        os.remove("invoice_database.db")
        print(f"Database dropped successfully.")
    except FileNotFoundError:
        print(f"Database does not exist.")
    except Exception as e:
        print(f"Error dropping database: {e}")

def create_table(table_name, df):
    """Creates new table using dataframe if not exists, else replaces table"""
    conn = connect_to_db()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def drop_table(table_name):
    pass

def get_data(db_cursor, sql_query):
    pass

def get_db_schema():
    """Retrieve the schema (DDL) for a table from an SQLite database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    # code to retrieve schema context
    try:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schema_tuples = cursor.fetchall()
        # Extract the schemas from the tuples and join them into a single string
        schema = '\n'.join(x[0] for x in schema_tuples)
        return schema
    except sqlite3.Error as e:
        print(f"Error retrieving schema: {e}")
    finally:
        if conn:
            conn.close()

# Setting up Invoice Database
def create_invoice_db():
    conn = sqlite3.connect('invoice_database.db')
    c = conn.cursor()

    # Create a Invoice Fact Table
    c.execute('''
        CREATE TABLE FT_INVOICE(
            INVOICE_ID INTEGER,
            DEPT_ID INTEGER,
            PRODUCT_ID INTEGER,
            PAYMENT_AMOUNT INTEGER,
            VENDOR_ID INTEGER,
            EMPLOYEE_ID INTEGER
        )''')
    
    # Create a Employee Lookup Table
    c.execute('''
        CREATE TABLE LU_EMPLOYEE(
            EMPLOYEE_ID INTEGER,
            EMPLOYEE_NAME TEXT
        )''')
    
    # Create a Product Lookup Table
    c.execute('''
        CREATE TABLE LU_PRODUCT(
            PRODUCT_ID INTEGER,
            PRODUCT_NAME TEXT
        )''')

    # Create a Vendor Lookup Table
    c.execute('''
        CREATE TABLE LU_VENDOR(
            VENDOR_ID INTEGER,
            VENDOR_FULLNAME TEXT,
            COUNTRY_CODE TEXT
        )''')

    # Create a Department Lookup Table
    c.execute('''
        CREATE TABLE LU_DEPARTMENT(
            DEPT_ID INTEGER,
            DEPT_NAME TEXT
        )''')
    
    # Populate FT_INVOICE
    with open('raw_datasets/ft_invoice.csv', 'r') as fin:
        # csv.DictReader uses the first line in the file as column headings by default
        dr = csv.DictReader(fin)
        fieldnames = dr.fieldnames  # This will get you the column names
        data = [tuple(i[field] for field in fieldnames) for i in dr]
    # Prepare the SQL query
    query = f"INSERT INTO FT_INVOICE({', '.join(fieldnames)}) VALUES ({', '.join('?' * len(fieldnames))})"
    c.executemany(query, data)

    # Populate LU_EMPLOYEE
    with open('raw_datasets/lu_employee.csv', 'r') as fin:
        # csv.DictReader uses the first line in the file as column headings by default
        dr = csv.DictReader(fin)
        fieldnames = dr.fieldnames  # This will get you the column names
        data = [tuple(i[field] for field in fieldnames) for i in dr]
    # Prepare the SQL query
    query = f"INSERT INTO LU_EMPLOYEE({', '.join(fieldnames)}) VALUES ({', '.join('?' * len(fieldnames))})"
    c.executemany(query, data)

    # Populate LU_VENDOR
    with open('raw_datasets/lu_vendor.csv', 'r') as fin:
        # csv.DictReader uses the first line in the file as column headings by default
        dr = csv.DictReader(fin)
        fieldnames = dr.fieldnames  # This will get you the column names
        data = [tuple(i[field] for field in fieldnames) for i in dr]
    # Prepare the SQL query
    query = f"INSERT INTO LU_VENDOR({', '.join(fieldnames)}) VALUES ({', '.join('?' * len(fieldnames))})"
    c.executemany(query, data)

    # Populate LU_DEPARTMENT
    with open('raw_datasets/lu_department.csv', 'r') as fin:
        # csv.DictReader uses the first line in the file as column headings by default
        dr = csv.DictReader(fin)
        fieldnames = dr.fieldnames  # This will get you the column names
        data = [tuple(i[field] for field in fieldnames) for i in dr]
    # Prepare the SQL query
    query = f"INSERT INTO LU_DEPARTMENT({', '.join(fieldnames)}) VALUES ({', '.join('?' * len(fieldnames))})"
    c.executemany(query, data)

    # Populate LU_PRODUCT
    with open('raw_datasets/lu_product.csv', 'r') as fin:
        # csv.DictReader uses the first line in the file as column headings by default
        dr = csv.DictReader(fin)
        fieldnames = dr.fieldnames  # This will get you the column names
        data = [tuple(i[field] for field in fieldnames) for i in dr]
    # Prepare the SQL query
    query = f"INSERT INTO LU_PRODUCT({', '.join(fieldnames)}) VALUES ({', '.join('?' * len(fieldnames))})"
    c.executemany(query, data)

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()