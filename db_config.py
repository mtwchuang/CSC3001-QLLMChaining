# libraries
import sqlite3
import os
import pandas as pd

# Sqlite Function Calls
def connect_to_db():
    """Creates sqlite database if not exists, else returns connection"""
    conn = sqlite3.connect('database.db')
    return conn

def drop_db():
    """Drop the SQLite database."""
    try:
        os.remove("database.db")
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

def get_db_schema():
    """Retrieve the schema (DDL) for a table from an SQLite database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    # code to retrieve schema context
    try:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schema = cursor.fetchone()[0]
        return schema
    except sqlite3.Error as e:
        print(f"Error retrieving schema: {e}")
    finally:
        if conn:
            conn.close()