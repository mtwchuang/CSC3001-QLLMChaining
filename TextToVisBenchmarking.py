# Local Libraries
import LLMConfiguration
import DatabaseConfiguration
# Others
import numpy as np
import time
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
# Ignore Visualizatin Generated
import matplotlib
matplotlib.use('Agg')

def load_easy_benchmark():
    """Loads benchmark dataset of 10 question-data pair of 100-250 tokens to turn into python visualization"""
    benchmark_data = [
        {
            # 204 tokens
            "question": "What are the total sales amounts for each employee in the Sales department?",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 5000}, {'EMPLOYEE_NAME': 'Jane Smith', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 7000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'EMPLOYEE_NAME': 'John Doe', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 5000}, {'EMPLOYEE_NAME': 'Jane Smith', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 7000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['EMPLOYEE_NAME'], df['TOTAL_SALES'], color='skyblue')
plt.xlabel('Employee Name')
plt.ylabel('Total Sales')
plt.title('Total Sales Amounts for Each Employee in the Sales Department')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 201 tokens
            "question": "What are the number of transactions handled by each employee in the Finance department?",
            "raw_data": "[{'EMPLOYEE_NAME': 'Alice Brown', 'DEPT_NAME': 'Finance', 'TRANSACTION_COUNT': 120}, {'EMPLOYEE_NAME': 'Bob White', 'DEPT_NAME': 'Finance', 'TRANSACTION_COUNT': 150}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'EMPLOYEE_NAME': 'Alice Brown', 'DEPT_NAME': 'Finance', 'TRANSACTION_COUNT': 120}, {'EMPLOYEE_NAME': 'Bob White', 'DEPT_NAME': 'Finance', 'TRANSACTION_COUNT': 150}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['EMPLOYEE_NAME'], df['TRANSACTION_COUNT'], color='skyblue')
plt.xlabel('Employee Name')
plt.ylabel('Transaction Count')
plt.title('Number of Transactions Handled by Each Employee in the Finance Department')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 210 tokens
            "question": "What is the distribution of product sales in different categories?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'SALES': 15000}, {'CATEGORY': 'Furniture', 'SALES': 8000}, {'CATEGORY': 'Clothing', 'SALES': 12000}, {'CATEGORY': 'Toys', 'SALES': 5000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'SALES': 15000}, {'CATEGORY': 'Furniture', 'SALES': 8000}, {'CATEGORY': 'Clothing', 'SALES': 12000}, {'CATEGORY': 'Toys', 'SALES': 5000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['CATEGORY'], df['SALES'], color='green')
plt.xlabel('Product Category')
plt.ylabel('Sales Amount')
plt.title('Distribution of Product Sales in Different Categories')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 250 tokens
            "question": "What are the average sales amounts per month for each product category?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'JAN': 5000, 'FEB': 4000, 'MAR': 3000}, {'CATEGORY': 'Furniture', 'JAN': 3000, 'FEB': 3500, 'MAR': 1500}, {'CATEGORY': 'Clothing', 'JAN': 4000, 'FEB': 3000, 'MAR': 5000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'JAN': 5000, 'FEB': 4000, 'MAR': 3000}, {'CATEGORY': 'Furniture', 'JAN': 3000, 'FEB': 3500, 'MAR': 1500}, {'CATEGORY': 'Clothing', 'JAN': 4000, 'FEB': 3000, 'MAR': 5000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Sales Amount')
plt.title('Average Sales Amounts per Month for Each Product Category')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 240 tokens
            "question": "What is the distribution of employee counts across different departments?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'EMPLOYEE_COUNT': 50}, {'DEPT_NAME': 'Marketing', 'EMPLOYEE_COUNT': 40}, {'DEPT_NAME': 'Engineering', 'EMPLOYEE_COUNT': 60}, {'DEPT_NAME': 'HR', 'EMPLOYEE_COUNT': 30}, {'DEPT_NAME': 'Finance', 'EMPLOYEE_COUNT': 20}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'EMPLOYEE_COUNT': 50}, {'DEPT_NAME': 'Marketing', 'EMPLOYEE_COUNT': 40}, {'DEPT_NAME': 'Engineering', 'EMPLOYEE_COUNT': 60}, {'DEPT_NAME': 'HR', 'EMPLOYEE_COUNT': 30}, {'DEPT_NAME': 'Finance', 'EMPLOYEE_COUNT': 20}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.pie(df['EMPLOYEE_COUNT'], labels=df['DEPT_NAME'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Employee Counts Across Different Departments')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 215 tokens
            "question": "What are the total sales amounts for each region in the first quarter?",
            "raw_data": "[{'REGION': 'North', 'Q1_SALES': 10000}, {'REGION': 'South', 'Q1_SALES': 15000}, {'REGION': 'East', 'Q1_SALES': 20000}, {'REGION': 'West', 'Q1_SALES': 25000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'REGION': 'North', 'Q1_SALES': 10000}, {'REGION': 'South', 'Q1_SALES': 15000}, {'REGION': 'East', 'Q1_SALES': 20000}, {'REGION': 'West', 'Q1_SALES': 25000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['REGION'], df['Q1_SALES'], color='orange')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.title('Total Sales Amounts for Each Region in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 227 tokens
            "question": "What is the distribution of sales for each product category in the second quarter?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'Q2_SALES': 20000}, {'CATEGORY': 'Furniture', 'Q2_SALES': 10000}, {'CATEGORY': 'Clothing', 'Q2_SALES': 15000}, {'CATEGORY': 'Toys', 'Q2_SALES': 5000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'Q2_SALES': 20000}, {'CATEGORY': 'Furniture', 'Q2_SALES': 10000}, {'CATEGORY': 'Clothing', 'Q2_SALES': 15000}, {'CATEGORY': 'Toys', 'Q2_SALES': 5000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['CATEGORY'], df['Q2_SALES'], color='purple')
plt.xlabel('Product Category')
plt.ylabel('Sales Amount')
plt.title('Distribution of Sales for Each Product Category in the Second Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 246 tokens
            "question": "What is the average monthly revenue for each department in the last year?",
            "raw_data": "[{'DEPARTMENT': 'HR', 'JAN': 5000, 'FEB': 5500, 'MAR': 5200}, {'DEPARTMENT': 'IT', 'JAN': 10000, 'FEB': 9500, 'MAR': 9700}, {'DEPARTMENT': 'Finance', 'JAN': 8000, 'FEB': 8200, 'MAR': 8300}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPARTMENT': 'HR', 'JAN': 5000, 'FEB': 5500, 'MAR': 5200}, {'DEPARTMENT': 'IT', 'JAN': 10000, 'FEB': 9500, 'MAR': 9700}, {'DEPARTMENT': 'Finance', 'JAN': 8000, 'FEB': 8200, 'MAR': 8300}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPARTMENT').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Revenue')
plt.title('Average Monthly Revenue for Each Department in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 232 tokens
            "question": "What are the total sales amounts for each product category over the last year?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'TOTAL_SALES': 50000}, {'CATEGORY': 'Furniture', 'TOTAL_SALES': 30000}, {'CATEGORY': 'Clothing', 'TOTAL_SALES': 20000}, {'CATEGORY': 'Toys', 'TOTAL_SALES': 15000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'TOTAL_SALES': 50000}, {'CATEGORY': 'Furniture', 'TOTAL_SALES': 30000}, {'CATEGORY': 'Clothing', 'TOTAL_SALES': 20000}, {'CATEGORY': 'Toys', 'TOTAL_SALES': 15000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['CATEGORY'], df['TOTAL_SALES'], color='purple')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.title('Total Sales Amounts for Each Product Category Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 235 tokens
            "question": "What is the yearly revenue distribution among the different departments?",
            "raw_data": "[{'DEPARTMENT': 'Sales', 'YEARLY_REVENUE': 500000}, {'DEPARTMENT': 'Marketing', 'YEARLY_REVENUE': 300000}, {'DEPARTMENT': 'HR', 'YEARLY_REVENUE': 200000}, {'DEPARTMENT': 'Finance', 'YEARLY_REVENUE': 400000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPARTMENT': 'Sales', 'YEARLY_REVENUE': 500000}, {'DEPARTMENT': 'Marketing', 'YEARLY_REVENUE': 300000}, {'DEPARTMENT': 'HR', 'YEARLY_REVENUE': 200000}, {'DEPARTMENT': 'Finance', 'YEARLY_REVENUE': 400000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.pie(df['YEARLY_REVENUE'], labels=df['DEPARTMENT'], autopct='%1.1f%%', startangle=140)
plt.title('Yearly Revenue Distribution Among Different Departments')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 227 tokens
            "question": "What are the total expenses amounts for each department in the first quarter?",
            "raw_data": "[{'DEPT': 'HR', 'Q1_EXPENSES': 12000}, {'DEPT': 'Finance', 'Q1_EXPENSES': 18000}, {'DEPT': 'IT', 'Q1_EXPENSES': 22000}, {'DEPT': 'Marketing', 'Q1_EXPENSES': 25000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT': 'HR', 'Q1_EXPENSES': 12000}, {'DEPT': 'Finance', 'Q1_EXPENSES': 18000}, {'DEPT': 'IT', 'Q1_EXPENSES': 22000}, {'DEPT': 'Marketing', 'Q1_EXPENSES': 25000}]
df = pd.DataFrame(data)

# Plot
df.set_index('DEPT').plot(kind='bar')
plt.xlabel('Department')
plt.ylabel('Expenses')
plt.title('Total Expenses Amounts for Each Department in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 229 tokens
            "question": "What are the total revenue amounts for each product in the first quarter?",
            "raw_data": "[{'PRODUCT': 'Product A', 'Q1_REVENUE': 30000}, {'PRODUCT': 'Product B', 'Q1_REVENUE': 35000}, {'PRODUCT': 'Product C', 'Q1_REVENUE': 40000}, {'PRODUCT': 'Product D', 'Q1_REVENUE': 45000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'PRODUCT': 'Product A', 'Q1_REVENUE': 30000}, {'PRODUCT': 'Product B', 'Q1_REVENUE': 35000}, {'PRODUCT': 'Product C', 'Q1_REVENUE': 40000}, {'PRODUCT': 'Product D', 'Q1_REVENUE': 45000}]
df = pd.DataFrame(data)

# Plot
df.set_index('PRODUCT').plot(kind='bar')
plt.xlabel('Product')
plt.ylabel('Revenue')
plt.title('Total Revenue Amounts for Each Product in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 227 tokens
            "question": "What are the total profit amounts for each branch in the first quarter?",
            "raw_data": "[{'BRANCH': 'Branch A', 'Q1_PROFIT': 5000}, {'BRANCH': 'Branch B', 'Q1_PROFIT': 10000}, {'BRANCH': 'Branch C', 'Q1_PROFIT': 15000}, {'BRANCH': 'Branch D', 'Q1_PROFIT': 20000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'BRANCH': 'Branch A', 'Q1_PROFIT': 5000}, {'BRANCH': 'Branch B', 'Q1_PROFIT': 10000}, {'BRANCH': 'Branch C', 'Q1_PROFIT': 15000}, {'BRANCH': 'Branch D', 'Q1_PROFIT': 20000}]
df = pd.DataFrame(data)

# Plot
df.set_index('BRANCH').plot(kind='bar')
plt.xlabel('Branch')
plt.ylabel('Profit')
plt.title('Total Profit Amounts for Each Branch in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 216 tokens
            "question": "What are the total cost amounts for each project in the first quarter?",
            "raw_data": "[{'PROJECT': 'Project X', 'Q1_COST': 15000}, {'PROJECT': 'Project Y', 'Q1_COST': 20000}, {'PROJECT': 'Project Z', 'Q1_COST': 25000}, {'PROJECT': 'Project W', 'Q1_COST': 30000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'PROJECT': 'Project X', 'Q1_COST': 15000}, {'PROJECT': 'Project Y', 'Q1_COST': 20000}, {'PROJECT': 'Project Z', 'Q1_COST': 25000}, {'PROJECT': 'Project W', 'Q1_COST': 30000}]
df = pd.DataFrame(data)

# Plot
df.set_index('PROJECT').plot(kind='bar')
plt.xlabel('Project')
plt.ylabel('Cost')
plt.title('Total Cost Amounts for Each Project in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 233 tokens
            "question": "What are the total investment amounts for each sector in the first quarter?",
            "raw_data": "[{'SECTOR': 'Sector A', 'Q1_INVESTMENT': 25000}, {'SECTOR': 'Sector B', 'Q1_INVESTMENT': 30000}, {'SECTOR': 'Sector C', 'Q1_INVESTMENT': 35000}, {'SECTOR': 'Sector D', 'Q1_INVESTMENT': 40000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'SECTOR': 'Sector A', 'Q1_INVESTMENT': 25000}, {'SECTOR': 'Sector B', 'Q1_INVESTMENT': 30000}, {'SECTOR': 'Sector C', 'Q1_INVESTMENT': 35000}, {'SECTOR': 'Sector D', 'Q1_INVESTMENT': 40000}]
df = pd.DataFrame(data)

# Plot
df.set_index('SECTOR').plot(kind='bar')
plt.xlabel('Sector')
plt.ylabel('Investment')
plt.title('Total Investment Amounts for Each Sector in the First Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        }
    ]
    return benchmark_data

# 15 medium testcases
def load_medium_benchmark():
    """Loads benchmark dataset of 10 question-data pair of 250-500 tokens to turn into python visualization"""
    benchmark_data = [
        {
            # 361 tokens
            "question": "What is the monthly revenue trend for each product category over the last quarter?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'JAN_REVENUE': 15000, 'FEB_REVENUE': 18000, 'MAR_REVENUE': 20000}, {'CATEGORY': 'Furniture', 'JAN_REVENUE': 8000, 'FEB_REVENUE': 7000, 'MAR_REVENUE': 7500}, {'CATEGORY': 'Clothing', 'JAN_REVENUE': 12000, 'FEB_REVENUE': 15000, 'MAR_REVENUE': 13000}, {'CATEGORY': 'Toys', 'JAN_REVENUE': 5000, 'FEB_REVENUE': 6000, 'MAR_REVENUE': 7000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'JAN_REVENUE': 15000, 'FEB_REVENUE': 18000, 'MAR_REVENUE': 20000}, {'CATEGORY': 'Furniture', 'JAN_REVENUE': 8000, 'FEB_REVENUE': 7000, 'MAR_REVENUE': 7500}, {'CATEGORY': 'Clothing', 'JAN_REVENUE': 12000, 'FEB_REVENUE': 15000, 'MAR_REVENUE': 13000}, {'CATEGORY': 'Toys', 'JAN_REVENUE': 5000, 'FEB_REVENUE': 6000, 'MAR_REVENUE': 7000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Monthly Revenue Trend for Each Product Category Over the Last Quarter')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 362 tokens
            "question": "What are the total expenses and revenues for each department in the last fiscal year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'TOTAL_EXPENSES': 30000, 'TOTAL_REVENUES': 500000}, {'DEPT_NAME': 'Marketing', 'TOTAL_EXPENSES': 20000, 'TOTAL_REVENUES': 400000}, {'DEPT_NAME': 'Engineering', 'TOTAL_EXPENSES': 25000, 'TOTAL_REVENUES': 450000}, {'DEPT_NAME': 'HR', 'TOTAL_EXPENSES': 15000, 'TOTAL_REVENUES': 350000}, {'DEPT_NAME': 'Finance', 'TOTAL_EXPENSES': 10000, 'TOTAL_REVENUES': 300000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'TOTAL_EXPENSES': 30000, 'TOTAL_REVENUES': 500000}, {'DEPT_NAME': 'Marketing', 'TOTAL_EXPENSES': 20000, 'TOTAL_REVENUES': 400000}, {'DEPT_NAME': 'Engineering', 'TOTAL_EXPENSES': 25000, 'TOTAL_REVENUES': 450000}, {'DEPT_NAME': 'HR', 'TOTAL_EXPENSES': 15000, 'TOTAL_REVENUES': 350000}, {'DEPT_NAME': 'Finance', 'TOTAL_EXPENSES': 10000, 'TOTAL_REVENUES': 300000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.plot(kind='bar', x='DEPT_NAME')
plt.xlabel('Department Name')
plt.ylabel('Amount')
plt.title('Total Expenses and Revenues for Each Department in the Last Fiscal Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 330 tokens
            "question": "What are the quarterly profits for each department in the last fiscal year?",
            "raw_data": "[{'DEPARTMENT': 'Sales', 'Q1_PROFIT': 10000, 'Q2_PROFIT': 15000, 'Q3_PROFIT': 20000, 'Q4_PROFIT': 25000}, {'DEPARTMENT': 'Marketing', 'Q1_PROFIT': 5000, 'Q2_PROFIT': 7000, 'Q3_PROFIT': 8000, 'Q4_PROFIT': 10000}, {'DEPARTMENT': 'HR', 'Q1_PROFIT': 3000, 'Q2_PROFIT': 4000, 'Q3_PROFIT': 5000, 'Q4_PROFIT': 6000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPARTMENT': 'Sales', 'Q1_PROFIT': 10000, 'Q2_PROFIT': 15000, 'Q3_PROFIT': 20000, 'Q4_PROFIT': 25000}, {'DEPARTMENT': 'Marketing', 'Q1_PROFIT': 5000, 'Q2_PROFIT': 7000, 'Q3_PROFIT': 8000, 'Q4_PROFIT': 10000}, {'DEPARTMENT': 'HR', 'Q1_PROFIT': 3000, 'Q2_PROFIT': 4000, 'Q3_PROFIT': 5000, 'Q4_PROFIT': 6000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPARTMENT').T.plot(kind='bar')
plt.xlabel('Quarter')
plt.ylabel('Profit Amount')
plt.title('Quarterly Profits for Each Department in the Last Fiscal Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 340 tokens
            "question": "What is the monthly expense trend for each department over the last six months?",
            "raw_data": "[{'DEPARTMENT': 'Sales', 'JAN': 8000, 'FEB': 9000, 'MAR': 8500, 'APR': 9500, 'MAY': 10000, 'JUN': 10500}, {'DEPARTMENT': 'IT', 'JAN': 7000, 'FEB': 7200, 'MAR': 7400, 'APR': 7600, 'MAY': 7800, 'JUN': 8000}, {'DEPARTMENT': 'Finance', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPARTMENT': 'Sales', 'JAN': 8000, 'FEB': 9000, 'MAR': 8500, 'APR': 9500, 'MAY': 10000, 'JUN': 10500}, {'DEPARTMENT': 'IT', 'JAN': 7000, 'FEB': 7200, 'MAR': 7400, 'APR': 7600, 'MAY': 7800, 'JUN': 8000}, {'DEPARTMENT': 'Finance', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPARTMENT').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Expense Amount')
plt.title('Monthly Expense Trend for Each Department Over the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 335 tokens
            "question": "What is the monthly sales trend for the top three products?",
            "raw_data": "[{'PRODUCT': 'Product A', 'JAN': 4000, 'FEB': 5000, 'MAR': 4500, 'APR': 4800, 'MAY': 5200, 'JUN': 6000}, {'PRODUCT': 'Product B', 'JAN': 3000, 'FEB': 3200, 'MAR': 3300, 'APR': 3500, 'MAY': 3700, 'JUN': 4000}, {'PRODUCT': 'Product C', 'JAN': 2000, 'FEB': 2100, 'MAR': 2200, 'APR': 2300, 'MAY': 2400, 'JUN': 2500}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'PRODUCT': 'Product A', 'JAN': 4000, 'FEB': 5000, 'MAR': 4500, 'APR': 4800, 'MAY': 5200, 'JUN': 6000}, {'PRODUCT': 'Product B', 'JAN': 3000, 'FEB': 3200, 'MAR': 3300, 'APR': 3500, 'MAY': 3700, 'JUN': 4000}, {'PRODUCT': 'Product C', 'JAN': 2000, 'FEB': 2100, 'MAR': 2200, 'APR': 2300, 'MAY': 2400, 'JUN': 2500}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('PRODUCT').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Sales Amount')
plt.title('Monthly Sales Trend for the Top Three Products')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 332 tokens
            "question": "What are the average monthly expenses for each department?",
            "raw_data": "[{'DEPARTMENT': 'Sales', 'JAN': 3000, 'FEB': 3200, 'MAR': 3100, 'APR': 3300, 'MAY': 3400, 'JUN': 3500}, {'DEPARTMENT': 'Marketing', 'JAN': 4000, 'FEB': 4100, 'MAR': 4200, 'APR': 4300, 'MAY': 4400, 'JUN': 4500}, {'DEPARTMENT': 'HR', 'JAN': 2000, 'FEB': 2100, 'MAR': 2200, 'APR': 2300, 'MAY': 2400, 'JUN': 2500}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPARTMENT': 'Sales', 'JAN': 3000, 'FEB': 3200, 'MAR': 3100, 'APR': 3300, 'MAY': 3400, 'JUN': 3500}, {'DEPARTMENT': 'Marketing', 'JAN': 4000, 'FEB': 4100, 'MAR': 4200, 'APR': 4300, 'MAY': 4400, 'JUN': 4500}, {'DEPARTMENT': 'HR', 'JAN': 2000, 'FEB': 2100, 'MAR': 2200, 'APR': 2300, 'MAY': 2400, 'JUN': 2500}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPARTMENT').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Average Monthly Expenses')
plt.title('Average Monthly Expenses for Each Department')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 398 tokens
            "question": "What is the quarterly sales trend for each product category in the current year?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'Q1_SALES': 40000, 'Q2_SALES': 45000, 'Q3_SALES': 48000, 'Q4_SALES': 52000}, {'CATEGORY': 'Furniture', 'Q1_SALES': 15000, 'Q2_SALES': 18000, 'Q3_SALES': 20000, 'Q4_SALES': 21000}, {'CATEGORY': 'Clothing', 'Q1_SALES': 25000, 'Q2_SALES': 27000, 'Q3_SALES': 29000, 'Q4_SALES': 30000}, {'CATEGORY': 'Toys', 'Q1_SALES': 10000, 'Q2_SALES': 12000, 'Q3_SALES': 13000, 'Q4_SALES': 15000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'Q1_SALES': 40000, 'Q2_SALES': 45000, 'Q3_SALES': 48000, 'Q4_SALES': 52000}, {'CATEGORY': 'Furniture', 'Q1_SALES': 15000, 'Q2_SALES': 18000, 'Q3_SALES': 20000, 'Q4_SALES': 21000}, {'CATEGORY': 'Clothing', 'Q1_SALES': 25000, 'Q2_SALES': 27000, 'Q3_SALES': 29000, 'Q4_SALES': 30000}, {'CATEGORY': 'Toys', 'Q1_SALES': 10000, 'Q2_SALES': 12000, 'Q3_SALES': 13000, 'Q4_SALES': 15000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='line')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.title('Quarterly Sales Trend for Each Product Category in the Current Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 407 tokens
            "question": "What are the quarterly profit margins for each department in the last year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'Q1_PROFIT': 20000, 'Q2_PROFIT': 22000, 'Q3_PROFIT': 25000, 'Q4_PROFIT': 27000}, {'DEPT_NAME': 'Marketing', 'Q1_PROFIT': 10000, 'Q2_PROFIT': 12000, 'Q3_PROFIT': 13000, 'Q4_PROFIT': 14000}, {'DEPT_NAME': 'Engineering', 'Q1_PROFIT': 15000, 'Q2_PROFIT': 17000, 'Q3_PROFIT': 18000, 'Q4_PROFIT': 19000}, {'DEPT_NAME': 'HR', 'Q1_PROFIT': 8000, 'Q2_PROFIT': 9000, 'Q3_PROFIT': 10000, 'Q4_PROFIT': 11000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'Q1_PROFIT': 20000, 'Q2_PROFIT': 22000, 'Q3_PROFIT': 25000, 'Q4_PROFIT': 27000}, {'DEPT_NAME': 'Marketing', 'Q1_PROFIT': 10000, 'Q2_PROFIT': 12000, 'Q3_PROFIT': 13000, 'Q4_PROFIT': 14000}, {'DEPT_NAME': 'Engineering', 'Q1_PROFIT': 15000, 'Q2_PROFIT': 17000, 'Q3_PROFIT': 18000, 'Q4_PROFIT': 19000}, {'DEPT_NAME': 'HR', 'Q1_PROFIT': 8000, 'Q2_PROFIT': 9000, 'Q3_PROFIT': 10000, 'Q4_PROFIT': 11000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Quarter')
plt.ylabel('Profit')
plt.title('Quarterly Profit Margins for Each Department in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 422 tokens
            "question": "What are the quarterly revenue and expense trends for each product category over the past year?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'Q1_REVENUE': 30000, 'Q1_EXPENSES': 15000, 'Q2_REVENUE': 35000, 'Q2_EXPENSES': 16000, 'Q3_REVENUE': 40000, 'Q3_EXPENSES': 17000, 'Q4_REVENUE': 45000, 'Q4_EXPENSES': 18000}, {'CATEGORY': 'Furniture', 'Q1_REVENUE': 20000, 'Q1_EXPENSES': 10000, 'Q2_REVENUE': 25000, 'Q2_EXPENSES': 11000, 'Q3_REVENUE': 30000, 'Q3_EXPENSES': 12000, 'Q4_REVENUE': 35000, 'Q4_EXPENSES': 13000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'Q1_REVENUE': 30000, 'Q1_EXPENSES': 15000, 'Q2_REVENUE': 35000, 'Q2_EXPENSES': 16000, 'Q3_REVENUE': 40000, 'Q3_EXPENSES': 17000, 'Q4_REVENUE': 45000, 'Q4_EXPENSES': 18000}, {'CATEGORY': 'Furniture', 'Q1_REVENUE': 20000, 'Q1_EXPENSES': 10000, 'Q2_REVENUE': 25000, 'Q2_EXPENSES': 11000, 'Q3_REVENUE': 30000, 'Q3_EXPENSES': 12000, 'Q4_REVENUE': 35000, 'Q4_EXPENSES': 13000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='bar')
plt.xlabel('Quarter')
plt.ylabel('Amount')
plt.title('Quarterly Revenue and Expense Trends for Each Product Category Over the Past Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 489 tokens
            "question": "What are the monthly transaction counts and average transaction values for each department in the last year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_TRANSACTIONS': 150, 'JAN_AVG_VALUE': 300, 'FEB_TRANSACTIONS': 160, 'FEB_AVG_VALUE': 320, 'MAR_TRANSACTIONS': 170, 'MAR_AVG_VALUE': 340, 'APR_TRANSACTIONS': 180, 'APR_AVG_VALUE': 360, 'MAY_TRANSACTIONS': 190, 'MAY_AVG_VALUE': 380, 'JUN_TRANSACTIONS': 200, 'JUN_AVG_VALUE': 400}, {'DEPT_NAME': 'Marketing', 'JAN_TRANSACTIONS': 100, 'JAN_AVG_VALUE': 200, 'FEB_TRANSACTIONS': 110, 'FEB_AVG_VALUE': 220, 'MAR_TRANSACTIONS': 120, 'MAR_AVG_VALUE': 240, 'APR_TRANSACTIONS': 130, 'APR_AVG_VALUE': 260, 'MAY_TRANSACTIONS': 140, 'MAY_AVG_VALUE': 280, 'JUN_TRANSACTIONS': 150, 'JUN_AVG_VALUE': 300}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_TRANSACTIONS': 150, 'JAN_AVG_VALUE': 300, 'FEB_TRANSACTIONS': 160, 'FEB_AVG_VALUE': 320, 'MAR_TRANSACTIONS': 170, 'MAR_AVG_VALUE': 340, 'APR_TRANSACTIONS': 180, 'APR_AVG_VALUE': 360, 'MAY_TRANSACTIONS': 190, 'MAY_AVG_VALUE': 380, 'JUN_TRANSACTIONS': 200, 'JUN_AVG_VALUE': 400}, {'DEPT_NAME': 'Marketing', 'JAN_TRANSACTIONS': 100, 'JAN_AVG_VALUE': 200, 'FEB_TRANSACTIONS': 110, 'FEB_AVG_VALUE': 220, 'MAR_TRANSACTIONS': 120, 'MAR_AVG_VALUE': 240, 'APR_TRANSACTIONS': 130, 'APR_AVG_VALUE': 260, 'MAY_TRANSACTIONS': 140, 'MAY_AVG_VALUE': 280, 'JUN_TRANSACTIONS': 150, 'JUN_AVG_VALUE': 300}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Transaction Counts and Average Values')
plt.title('Monthly Transaction Counts and Average Transaction Values for Each Department in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 396 tokens
            "question": "What are the quarterly sales and expenses for each department in the last year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'Q1_SALES': 40000, 'Q1_EXPENSES': 10000, 'Q2_SALES': 45000, 'Q2_EXPENSES': 12000, 'Q3_SALES': 47000, 'Q3_EXPENSES': 15000, 'Q4_SALES': 50000, 'Q4_EXPENSES': 20000}, {'DEPT_NAME': 'Marketing', 'Q1_SALES': 20000, 'Q1_EXPENSES': 5000, 'Q2_SALES': 22000, 'Q2_EXPENSES': 7000, 'Q3_SALES': 25000, 'Q3_EXPENSES': 8000, 'Q4_SALES': 27000, 'Q4_EXPENSES': 10000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'Q1_SALES': 40000, 'Q1_EXPENSES': 10000, 'Q2_SALES': 45000, 'Q2_EXPENSES': 12000, 'Q3_SALES': 47000, 'Q3_EXPENSES': 15000, 'Q4_SALES': 50000, 'Q4_EXPENSES': 20000}, {'DEPT_NAME': 'Marketing', 'Q1_SALES': 20000, 'Q1_EXPENSES': 5000, 'Q2_SALES': 22000, 'Q2_EXPENSES': 7000, 'Q3_SALES': 25000, 'Q3_EXPENSES': 8000, 'Q4_SALES': 27000, 'Q4_EXPENSES': 10000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='bar')
plt.xlabel('Quarter')
plt.ylabel('Amount')
plt.title('Quarterly Sales and Expenses for Each Department in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 332 tokens
            "question": "What is the monthly revenue trend for the top three services?",
            "raw_data": "[{'SERVICE': 'Service A', 'JAN': 5000, 'FEB': 5200, 'MAR': 5300, 'APR': 5400, 'MAY': 5500, 'JUN': 5600}, {'SERVICE': 'Service B', 'JAN': 4000, 'FEB': 4100, 'MAR': 4200, 'APR': 4300, 'MAY': 4400, 'JUN': 4500}, {'SERVICE': 'Service C', 'JAN': 3000, 'FEB': 3100, 'MAR': 3200, 'APR': 3300, 'MAY': 3400, 'JUN': 3500}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'SERVICE': 'Service A', 'JAN': 5000, 'FEB': 5200, 'MAR': 5300, 'APR': 5400, 'MAY': 5500, 'JUN': 5600},
        {'SERVICE': 'Service B', 'JAN': 4000, 'FEB': 4100, 'MAR': 4200, 'APR': 4300, 'MAY': 4400, 'JUN': 4500},
        {'SERVICE': 'Service C', 'JAN': 3000, 'FEB': 3100, 'MAR': 3200, 'APR': 3300, 'MAY': 3400, 'JUN': 3500}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('SERVICE').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Monthly Revenue Trend for the Top Three Services')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 332 tokens
            "question": "What is the monthly profit trend for the top three regions?",
            "raw_data": "[{'REGION': 'Region A', 'JAN': 7000, 'FEB': 7200, 'MAR': 7500, 'APR': 7800, 'MAY': 8000, 'JUN': 8200}, {'REGION': 'Region B', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000}, {'REGION': 'Region C', 'JAN': 5000, 'FEB': 5200, 'MAR': 5400, 'APR': 5600, 'MAY': 5800, 'JUN': 6000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'REGION': 'Region A', 'JAN': 7000, 'FEB': 7200, 'MAR': 7500, 'APR': 7800, 'MAY': 8000, 'JUN': 8200},
        {'REGION': 'Region B', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000},
        {'REGION': 'Region C', 'JAN': 5000, 'FEB': 5200, 'MAR': 5400, 'APR': 5600, 'MAY': 5800, 'JUN': 6000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('REGION').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Profit')
plt.title('Monthly Profit Trend for the Top Three Regions')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 329 tokens
            "question": "What is the monthly customer acquisition trend for the top three campaigns?",
            "raw_data": "[{'CAMPAIGN': 'Campaign A', 'JAN': 300, 'FEB': 320, 'MAR': 340, 'APR': 360, 'MAY': 380, 'JUN': 400}, {'CAMPAIGN': 'Campaign B', 'JAN': 250, 'FEB': 270, 'MAR': 290, 'APR': 310, 'MAY': 330, 'JUN': 350}, {'CAMPAIGN': 'Campaign C', 'JAN': 200, 'FEB': 220, 'MAR': 240, 'APR': 260, 'MAY': 280, 'JUN': 300}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CAMPAIGN': 'Campaign A', 'JAN': 300, 'FEB': 320, 'MAR': 340, 'APR': 360, 'MAY': 380, 'JUN': 400},
        {'CAMPAIGN': 'Campaign B', 'JAN': 250, 'FEB': 270, 'MAR': 290, 'APR': 310, 'MAY': 330, 'JUN': 350},
        {'CAMPAIGN': 'Campaign C', 'JAN': 200, 'FEB': 220, 'MAR': 240, 'APR': 260, 'MAY': 280, 'JUN': 300}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CAMPAIGN').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Customer Acquisition')
plt.title('Monthly Customer Acquisition Trend for the Top Three Campaigns')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 334 tokens
            "question": "What is the monthly expenditure trend for the top three projects?",
            "raw_data": "[{'PROJECT': 'Project X', 'JAN': 8000, 'FEB': 8200, 'MAR': 8400, 'APR': 8600, 'MAY': 8800, 'JUN': 9000}, {'PROJECT': 'Project Y', 'JAN': 7000, 'FEB': 7200, 'MAR': 7400, 'APR': 7600, 'MAY': 7800, 'JUN': 8000}, {'PROJECT': 'Project Z', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'PROJECT': 'Project X', 'JAN': 8000, 'FEB': 8200, 'MAR': 8400, 'APR': 8600, 'MAY': 8800, 'JUN': 9000},
        {'PROJECT': 'Project Y', 'JAN': 7000, 'FEB': 7200, 'MAR': 7400, 'APR': 7600, 'MAY': 7800, 'JUN': 8000},
        {'PROJECT': 'Project Z', 'JAN': 6000, 'FEB': 6200, 'MAR': 6400, 'APR': 6600, 'MAY': 6800, 'JUN': 7000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('PROJECT').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Expenditure')
plt.title('Monthly Expenditure Trend for the Top Three Projects')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        }
    ]
    return benchmark_data

# 15 hard testcases
def load_hard_benchmark():
    """Loads benchmark dataset of 10 question-data pair of 500-650 tokens to turn into python visualization"""
    benchmark = [
        {
            # 539 tokens
            "question": "What are the monthly expenses and revenues for each department in the first half of the year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_EXPENSES': 3000, 'JAN_REVENUE': 50000, 'FEB_EXPENSES': 3200, 'FEB_REVENUE': 52000, 'MAR_EXPENSES': 3100, 'MAR_REVENUE': 53000, 'APR_EXPENSES': 3300, 'APR_REVENUE': 54000, 'MAY_EXPENSES': 3400, 'MAY_REVENUE': 55000, 'JUN_EXPENSES': 3500, 'JUN_REVENUE': 56000}, {'DEPT_NAME': 'Marketing', 'JAN_EXPENSES': 2000, 'JAN_REVENUE': 30000, 'FEB_EXPENSES': 2200, 'FEB_REVENUE': 31000, 'MAR_EXPENSES': 2100, 'MAR_REVENUE': 32000, 'APR_EXPENSES': 2300, 'APR_REVENUE': 33000, 'MAY_EXPENSES': 2400, 'MAY_REVENUE': 34000, 'JUN_EXPENSES': 2500, 'JUN_REVENUE': 35000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_EXPENSES': 3000, 'JAN_REVENUE': 50000, 'FEB_EXPENSES': 3200, 'FEB_REVENUE': 52000, 'MAR_EXPENSES': 3100, 'MAR_REVENUE': 53000, 'APR_EXPENSES': 3300, 'APR_REVENUE': 54000, 'MAY_EXPENSES': 3400, 'MAY_REVENUE': 55000, 'JUN_EXPENSES': 3500, 'JUN_REVENUE': 56000}, {'DEPT_NAME': 'Marketing', 'JAN_EXPENSES': 2000, 'JAN_REVENUE': 30000, 'FEB_EXPENSES': 2200, 'FEB_REVENUE': 31000, 'MAR_EXPENSES': 2100, 'MAR_REVENUE': 32000, 'APR_EXPENSES': 2300, 'APR_REVENUE': 33000, 'MAY_EXPENSES': 2400, 'MAY_REVENUE': 34000, 'JUN_EXPENSES': 2500, 'JUN_REVENUE': 35000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Monthly Expenses and Revenues for Each Department in the First Half of the Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 525 tokens
            "question": "What are the monthly expenses and profits for each department over the last six months?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_EXPENSES': 3000, 'JAN_PROFIT': 20000, 'FEB_EXPENSES': 3200, 'FEB_PROFIT': 22000, 'MAR_EXPENSES': 3100, 'MAR_PROFIT': 25000, 'APR_EXPENSES': 3300, 'APR_PROFIT': 27000, 'MAY_EXPENSES': 3400, 'MAY_PROFIT': 29000, 'JUN_EXPENSES': 3500, 'JUN_PROFIT': 30000}, {'DEPT_NAME': 'Marketing', 'JAN_EXPENSES': 2000, 'JAN_PROFIT': 10000, 'FEB_EXPENSES': 2200, 'FEB_PROFIT': 12000, 'MAR_EXPENSES': 2100, 'MAR_PROFIT': 13000, 'APR_EXPENSES': 2300, 'APR_PROFIT': 14000, 'MAY_EXPENSES': 2400, 'MAY_PROFIT': 15000, 'JUN_EXPENSES': 2500, 'JUN_PROFIT': 16000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_EXPENSES': 3000, 'JAN_PROFIT': 20000, 'FEB_EXPENSES': 3200, 'FEB_PROFIT': 22000, 'MAR_EXPENSES': 3100, 'MAR_PROFIT': 25000, 'APR_EXPENSES': 3300, 'APR_PROFIT': 27000, 'MAY_EXPENSES': 3400, 'MAY_PROFIT': 29000, 'JUN_EXPENSES': 3500, 'JUN_PROFIT': 30000}, {'DEPT_NAME': 'Marketing', 'JAN_EXPENSES': 2000, 'JAN_PROFIT': 10000, 'FEB_EXPENSES': 2200, 'FEB_PROFIT': 12000, 'MAR_EXPENSES': 2100, 'MAR_PROFIT': 13000, 'APR_EXPENSES': 2300, 'APR_PROFIT': 14000, 'MAY_EXPENSES': 2400, 'MAY_PROFIT': 15000, 'JUN_EXPENSES': 2500, 'JUN_PROFIT': 16000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Monthly Expenses and Profits for Each Department Over the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 502 tokens
            "question": "What is the yearly sales and profit margin trend for each department over the last three years?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'YEAR_1_SALES': 150000, 'YEAR_1_PROFIT_MARGIN': 0.2, 'YEAR_2_SALES': 160000, 'YEAR_2_PROFIT_MARGIN': 0.22, 'YEAR_3_SALES': 170000, 'YEAR_3_PROFIT_MARGIN': 0.24}, {'DEPT_NAME': 'Marketing', 'YEAR_1_SALES': 80000, 'YEAR_1_PROFIT_MARGIN': 0.15, 'YEAR_2_SALES': 85000, 'YEAR_2_PROFIT_MARGIN': 0.17, 'YEAR_3_SALES': 90000, 'YEAR_3_PROFIT_MARGIN': 0.18}, {'DEPT_NAME': 'Engineering', 'YEAR_1_SALES': 120000, 'YEAR_1_PROFIT_MARGIN': 0.25, 'YEAR_2_SALES': 130000, 'YEAR_2_PROFIT_MARGIN': 0.27, 'YEAR_3_SALES': 140000, 'YEAR_3_PROFIT_MARGIN': 0.29}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'YEAR_1_SALES': 150000, 'YEAR_1_PROFIT_MARGIN': 0.2, 'YEAR_2_SALES': 160000, 'YEAR_2_PROFIT_MARGIN': 0.22, 'YEAR_3_SALES': 170000, 'YEAR_3_PROFIT_MARGIN': 0.24}, {'DEPT_NAME': 'Marketing', 'YEAR_1_SALES': 80000, 'YEAR_1_PROFIT_MARGIN': 0.15, 'YEAR_2_SALES': 85000, 'YEAR_2_PROFIT_MARGIN': 0.17, 'YEAR_3_SALES': 90000, 'YEAR_3_PROFIT_MARGIN': 0.18}, {'DEPT_NAME': 'Engineering', 'YEAR_1_SALES': 120000, 'YEAR_1_PROFIT_MARGIN': 0.25, 'YEAR_2_SALES': 130000, 'YEAR_2_PROFIT_MARGIN': 0.27, 'YEAR_3_SALES': 140000, 'YEAR_3_PROFIT_MARGIN': 0.29}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Sales and Profit Margin')
plt.title('Yearly Sales and Profit Margin Trend for Each Department Over the Last Three Years')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 537 tokens
            "question": "What are the monthly expenses and revenues for each product category in the last six months?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'JAN_EXPENSES': 5000, 'JAN_REVENUE': 15000, 'FEB_EXPENSES': 5200, 'FEB_REVENUE': 16000, 'MAR_EXPENSES': 5300, 'MAR_REVENUE': 17000, 'APR_EXPENSES': 5400, 'APR_REVENUE': 18000, 'MAY_EXPENSES': 5500, 'MAY_REVENUE': 19000, 'JUN_EXPENSES': 5600, 'JUN_REVENUE': 20000}, {'CATEGORY': 'Furniture', 'JAN_EXPENSES': 3000, 'JAN_REVENUE': 8000, 'FEB_EXPENSES': 3200, 'FEB_REVENUE': 8500, 'MAR_EXPENSES': 3100, 'MAR_REVENUE': 9000, 'APR_EXPENSES': 3300, 'APR_REVENUE': 9500, 'MAY_EXPENSES': 3400, 'MAY_REVENUE': 10000, 'JUN_EXPENSES': 3500, 'JUN_REVENUE': 10500}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'JAN_EXPENSES': 5000, 'JAN_REVENUE': 15000, 'FEB_EXPENSES': 5200, 'FEB_REVENUE': 16000, 'MAR_EXPENSES': 5300, 'MAR_REVENUE': 17000, 'APR_EXPENSES': 5400, 'APR_REVENUE': 18000, 'MAY_EXPENSES': 5500, 'MAY_REVENUE': 19000, 'JUN_EXPENSES': 5600, 'JUN_REVENUE': 20000}, {'CATEGORY': 'Furniture', 'JAN_EXPENSES': 3000, 'JAN_REVENUE': 8000, 'FEB_EXPENSES': 3200, 'FEB_REVENUE': 8500, 'MAR_EXPENSES': 3100, 'MAR_REVENUE': 9000, 'APR_EXPENSES': 3300, 'APR_REVENUE': 9500, 'MAY_EXPENSES': 3400, 'MAY_REVENUE': 10000, 'JUN_EXPENSES': 3500, 'JUN_REVENUE': 10500}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Monthly Expenses and Revenues for Each Product Category in the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 551 tokens
            "question": "What are the monthly revenue and expense trends for each department over the last six months?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 50000, 'JAN_EXPENSES': 30000, 'FEB_REVENUE': 52000, 'FEB_EXPENSES': 32000, 'MAR_REVENUE': 53000, 'MAR_EXPENSES': 31000, 'APR_REVENUE': 54000, 'APR_EXPENSES': 33000, 'MAY_REVENUE': 55000, 'MAY_EXPENSES': 34000, 'JUN_REVENUE': 56000, 'JUN_EXPENSES': 35000}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 30000, 'JAN_EXPENSES': 20000, 'FEB_REVENUE': 31000, 'FEB_EXPENSES': 22000, 'MAR_REVENUE': 32000, 'MAR_EXPENSES': 21000, 'APR_REVENUE': 33000, 'APR_EXPENSES': 23000, 'MAY_REVENUE': 34000, 'MAY_EXPENSES': 24000, 'JUN_REVENUE': 35000, 'JUN_EXPENSES': 25000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 50000, 'JAN_EXPENSES': 30000, 'FEB_REVENUE': 52000, 'FEB_EXPENSES': 32000, 'MAR_REVENUE': 53000, 'MAR_EXPENSES': 31000, 'APR_REVENUE': 54000, 'APR_EXPENSES': 33000, 'MAY_REVENUE': 55000, 'MAY_EXPENSES': 34000, 'JUN_REVENUE': 56000, 'JUN_EXPENSES': 35000}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 30000, 'JAN_EXPENSES': 20000, 'FEB_REVENUE': 31000, 'FEB_EXPENSES': 22000, 'MAR_REVENUE': 32000, 'MAR_EXPENSES': 21000, 'APR_REVENUE': 33000, 'APR_EXPENSES': 23000, 'MAY_REVENUE': 34000, 'MAY_EXPENSES': 24000, 'JUN_REVENUE': 35000, 'JUN_EXPENSES': 25000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Monthly Revenue and Expense Trends for Each Department Over the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 574 tokens
            "question": "What are the monthly revenue trends and profit margins for each department in the last year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 45000, 'JAN_PROFIT_MARGIN': 0.22, 'FEB_REVENUE': 47000, 'FEB_PROFIT_MARGIN': 0.24, 'MAR_REVENUE': 48000, 'MAR_PROFIT_MARGIN': 0.23, 'APR_REVENUE': 50000, 'APR_PROFIT_MARGIN': 0.25, 'MAY_REVENUE': 52000, 'MAY_PROFIT_MARGIN': 0.26, 'JUN_REVENUE': 53000, 'JUN_PROFIT_MARGIN': 0.27}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 30000, 'JAN_PROFIT_MARGIN': 0.2, 'FEB_REVENUE': 31000, 'FEB_PROFIT_MARGIN': 0.21, 'MAR_REVENUE': 32000, 'MAR_PROFIT_MARGIN': 0.22, 'APR_REVENUE': 33000, 'APR_PROFIT_MARGIN': 0.23, 'MAY_REVENUE': 34000, 'MAY_PROFIT_MARGIN': 0.24, 'JUN_REVENUE': 35000, 'JUN_PROFIT_MARGIN': 0.25}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 45000, 'JAN_PROFIT_MARGIN': 0.22, 'FEB_REVENUE': 47000, 'FEB_PROFIT_MARGIN': 0.24, 'MAR_REVENUE': 48000, 'MAR_PROFIT_MARGIN': 0.23, 'APR_REVENUE': 50000, 'APR_PROFIT_MARGIN': 0.25, 'MAY_REVENUE': 52000, 'MAY_PROFIT_MARGIN': 0.26, 'JUN_REVENUE': 53000, 'JUN_PROFIT_MARGIN': 0.27}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 30000, 'JAN_PROFIT_MARGIN': 0.2, 'FEB_REVENUE': 31000, 'FEB_PROFIT_MARGIN': 0.21, 'MAR_REVENUE': 32000, 'MAR_PROFIT_MARGIN': 0.22, 'APR_REVENUE': 33000, 'APR_PROFIT_MARGIN': 0.23, 'MAY_REVENUE': 34000, 'MAY_PROFIT_MARGIN': 0.24, 'JUN_REVENUE': 35000, 'JUN_PROFIT_MARGIN': 0.25}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Revenue and Profit Margins')
plt.title('Monthly Revenue Trends and Profit Margins for Each Department in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 596 tokens
            "question": "What are the quarterly expenses and profit margins for each product category over the last year?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.25, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.28, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.26, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.29}, {'CATEGORY': 'Furniture', 'Q1_EXPENSES': 10000, 'Q1_PROFIT_MARGIN': 0.2, 'Q2_EXPENSES': 12000, 'Q2_PROFIT_MARGIN': 0.22, 'Q3_EXPENSES': 13000, 'Q3_PROFIT_MARGIN': 0.23, 'Q4_EXPENSES': 14000, 'Q4_PROFIT_MARGIN': 0.24}, {'CATEGORY': 'Clothing', 'Q1_EXPENSES': 9000, 'Q1_PROFIT_MARGIN': 0.3, 'Q2_EXPENSES': 10000, 'Q2_PROFIT_MARGIN': 0.32, 'Q3_EXPENSES': 11000, 'Q3_PROFIT_MARGIN': 0.34, 'Q4_EXPENSES': 12000, 'Q4_PROFIT_MARGIN': 0.35}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.25, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.28, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.26, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.29}, {'CATEGORY': 'Furniture', 'Q1_EXPENSES': 10000, 'Q1_PROFIT_MARGIN': 0.2, 'Q2_EXPENSES': 12000, 'Q2_PROFIT_MARGIN': 0.22, 'Q3_EXPENSES': 13000, 'Q3_PROFIT_MARGIN': 0.23, 'Q4_EXPENSES': 14000, 'Q4_PROFIT_MARGIN': 0.24}, {'CATEGORY': 'Clothing', 'Q1_EXPENSES': 9000, 'Q1_PROFIT_MARGIN': 0.3, 'Q2_EXPENSES': 10000, 'Q2_PROFIT_MARGIN': 0.32, 'Q3_EXPENSES': 11000, 'Q3_PROFIT_MARGIN': 0.34, 'Q4_EXPENSES': 12000, 'Q4_PROFIT_MARGIN': 0.35}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='bar')
plt.xlabel('Quarter')
plt.ylabel('Expenses and Profit Margins')
plt.title('Quarterly Expenses and Profit Margins for Each Product Category Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 575 tokens
            "question": "What are the monthly expenses and profit margins for each product category in the last year?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'JAN_EXPENSES': 5000, 'JAN_PROFIT_MARGIN': 0.25, 'FEB_EXPENSES': 5200, 'FEB_PROFIT_MARGIN': 0.27, 'MAR_EXPENSES': 5300, 'MAR_PROFIT_MARGIN': 0.26, 'APR_EXPENSES': 5400, 'APR_PROFIT_MARGIN': 0.28, 'MAY_EXPENSES': 5500, 'MAY_PROFIT_MARGIN': 0.29, 'JUN_EXPENSES': 5600, 'JUN_PROFIT_MARGIN': 0.30}, {'CATEGORY': 'Furniture', 'JAN_EXPENSES': 3000, 'JAN_PROFIT_MARGIN': 0.15, 'FEB_EXPENSES': 3200, 'FEB_PROFIT_MARGIN': 0.17, 'MAR_EXPENSES': 3100, 'MAR_PROFIT_MARGIN': 0.16, 'APR_EXPENSES': 3300, 'APR_PROFIT_MARGIN': 0.18, 'MAY_EXPENSES': 3400, 'MAY_PROFIT_MARGIN': 0.19, 'JUN_EXPENSES': 3500, 'JUN_PROFIT_MARGIN': 0.20}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'CATEGORY': 'Electronics', 'JAN_EXPENSES': 5000, 'JAN_PROFIT_MARGIN': 0.25, 'FEB_EXPENSES': 5200, 'FEB_PROFIT_MARGIN': 0.27, 'MAR_EXPENSES': 5300, 'MAR_PROFIT_MARGIN': 0.26, 'APR_EXPENSES': 5400, 'APR_PROFIT_MARGIN': 0.28, 'MAY_EXPENSES': 5500, 'MAY_PROFIT_MARGIN': 0.29, 'JUN_EXPENSES': 5600, 'JUN_PROFIT_MARGIN': 0.30}, {'CATEGORY': 'Furniture', 'JAN_EXPENSES': 3000, 'JAN_PROFIT_MARGIN': 0.15, 'FEB_EXPENSES': 3200, 'FEB_PROFIT_MARGIN': 0.17, 'MAR_EXPENSES': 3100, 'MAR_PROFIT_MARGIN': 0.16, 'APR_EXPENSES': 3300, 'APR_PROFIT_MARGIN': 0.18, 'MAY_EXPENSES': 3400, 'MAY_PROFIT_MARGIN': 0.19, 'JUN_EXPENSES': 3500, 'JUN_PROFIT_MARGIN': 0.20}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('CATEGORY').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Expenses and Profit Margins')
plt.title('Monthly Expenses and Profit Margins for Each Product Category in the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 541 tokens
            "question": "What is the monthly revenue and customer satisfaction trend for each service department over the last six months?",
            "raw_data": "[{'DEPT_NAME': 'Customer Service', 'JAN_REVENUE': 12000, 'JAN_SATISFACTION': 0.85, 'FEB_REVENUE': 13000, 'FEB_SATISFACTION': 0.86, 'MAR_REVENUE': 14000, 'MAR_SATISFACTION': 0.87, 'APR_REVENUE': 15000, 'APR_SATISFACTION': 0.88, 'MAY_REVENUE': 16000, 'MAY_SATISFACTION': 0.89, 'JUN_REVENUE': 17000, 'JUN_SATISFACTION': 0.9}, {'DEPT_NAME': 'Technical Support', 'JAN_REVENUE': 10000, 'JAN_SATISFACTION': 0.82, 'FEB_REVENUE': 11000, 'FEB_SATISFACTION': 0.83, 'MAR_REVENUE': 12000, 'MAR_SATISFACTION': 0.84, 'APR_REVENUE': 13000, 'APR_SATISFACTION': 0.85, 'MAY_REVENUE': 14000, 'MAY_SATISFACTION': 0.86, 'JUN_REVENUE': 15000, 'JUN_SATISFACTION': 0.87}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Customer Service', 'JAN_REVENUE': 12000, 'JAN_SATISFACTION': 0.85, 'FEB_REVENUE': 13000, 'FEB_SATISFACTION': 0.86, 'MAR_REVENUE': 14000, 'MAR_SATISFACTION': 0.87, 'APR_REVENUE': 15000, 'APR_SATISFACTION': 0.88, 'MAY_REVENUE': 16000, 'MAY_SATISFACTION': 0.89, 'JUN_REVENUE': 17000, 'JUN_SATISFACTION': 0.9}, {'DEPT_NAME': 'Technical Support', 'JAN_REVENUE': 10000, 'JAN_SATISFACTION': 0.82, 'FEB_REVENUE': 11000, 'FEB_SATISFACTION': 0.83, 'MAR_REVENUE': 12000, 'MAR_SATISFACTION': 0.84, 'APR_REVENUE': 13000, 'APR_SATISFACTION': 0.85, 'MAY_REVENUE': 14000, 'MAY_SATISFACTION': 0.86, 'JUN_REVENUE': 15000, 'JUN_SATISFACTION': 0.87}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(12, 8))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Revenue and Customer Satisfaction')
plt.title('Monthly Revenue and Customer Satisfaction Trend for Each Service Department Over the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 510 tokens
            "question": "What are the monthly revenue and profit trends for each department over the last six months?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 15000, 'JAN_PROFIT': 3000, 'FEB_REVENUE': 16000, 'FEB_PROFIT': 3200, 'MAR_REVENUE': 17000, 'MAR_PROFIT': 3400, 'APR_REVENUE': 18000, 'APR_PROFIT': 3600, 'MAY_REVENUE': 19000, 'MAY_PROFIT': 3800, 'JUN_REVENUE': 20000, 'JUN_PROFIT': 4000}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 8000, 'JAN_PROFIT': 1600, 'FEB_REVENUE': 8500, 'FEB_PROFIT': 1700, 'MAR_REVENUE': 9000, 'MAR_PROFIT': 1800, 'APR_REVENUE': 9500, 'APR_PROFIT': 1900, 'MAY_REVENUE': 10000, 'MAY_PROFIT': 2000, 'JUN_REVENUE': 10500, 'JUN_PROFIT': 2100}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Sales', 'JAN_REVENUE': 15000, 'JAN_PROFIT': 3000, 'FEB_REVENUE': 16000, 'FEB_PROFIT': 3200, 'MAR_REVENUE': 17000, 'MAR_PROFIT': 3400, 'APR_REVENUE': 18000, 'APR_PROFIT': 3600, 'MAY_REVENUE': 19000, 'MAY_PROFIT': 3800, 'JUN_REVENUE': 20000, 'JUN_PROFIT': 4000}, {'DEPT_NAME': 'Marketing', 'JAN_REVENUE': 8000, 'JAN_PROFIT': 1600, 'FEB_REVENUE': 8500, 'FEB_PROFIT': 1700, 'MAR_REVENUE': 9000, 'MAR_PROFIT': 1800, 'APR_REVENUE': 9500, 'APR_PROFIT': 1900, 'MAY_REVENUE': 10000, 'MAY_PROFIT': 2000, 'JUN_REVENUE': 10500, 'JUN_PROFIT': 2100}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(12, 8))
df.set_index('DEPT_NAME').T.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Revenue and Profit')
plt.title('Monthly Revenue and Profit Trends for Each Department Over the Last Six Months')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 522 tokens
            "question": "What are the monthly profit and expenses trends for the IT department over the last year?",
            "raw_data": "[{'DEPT_NAME': 'IT', 'JAN_PROFIT': 15000, 'JAN_EXPENSES': 5000, 'FEB_PROFIT': 16000, 'FEB_EXPENSES': 6000, 'MAR_PROFIT': 17000, 'MAR_EXPENSES': 7000, 'APR_PROFIT': 18000, 'APR_EXPENSES': 8000, 'MAY_PROFIT': 19000, 'MAY_EXPENSES': 9000, 'JUN_PROFIT': 20000, 'JUN_EXPENSES': 10000, 'JUL_PROFIT': 21000, 'JUL_EXPENSES': 11000, 'AUG_PROFIT': 22000, 'AUG_EXPENSES': 12000, 'SEP_PROFIT': 23000, 'SEP_EXPENSES': 13000, 'OCT_PROFIT': 24000, 'OCT_EXPENSES': 14000, 'NOV_PROFIT': 25000, 'NOV_EXPENSES': 15000, 'DEC_PROFIT': 26000, 'DEC_EXPENSES': 16000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'IT', 'JAN_PROFIT': 15000, 'JAN_EXPENSES': 5000, 'FEB_PROFIT': 16000, 'FEB_EXPENSES': 6000, 'MAR_PROFIT': 17000, 'MAR_EXPENSES': 7000, 'APR_PROFIT': 18000, 'APR_EXPENSES': 8000, 'MAY_PROFIT': 19000, 'MAY_EXPENSES': 9000, 'JUN_PROFIT': 20000, 'JUN_EXPENSES': 10000, 'JUL_PROFIT': 21000, 'JUL_EXPENSES': 11000, 'AUG_PROFIT': 22000, 'AUG_EXPENSES': 12000, 'SEP_PROFIT': 23000, 'SEP_EXPENSES': 13000, 'OCT_PROFIT': 24000, 'OCT_EXPENSES': 14000, 'NOV_PROFIT': 25000, 'NOV_EXPENSES': 15000, 'DEC_PROFIT': 26000, 'DEC_EXPENSES': 16000}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='line')
abel('Amount')
plt.title('Monthly Profit and Expenses Trends for the IT Department Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()plt.xlabel('Month')
plt.yl
"""
        },
        {
            # 503 tokens
            "question": "What are the monthly revenues and profits for the HR department over the last year?",
            "raw_data": "[{'DEPT_NAME': 'HR', 'JAN_REVENUE': 10000, 'JAN_PROFIT': 3000, 'FEB_REVENUE': 11000, 'FEB_PROFIT': 3200, 'MAR_REVENUE': 12000, 'MAR_PROFIT': 3500, 'APR_REVENUE': 13000, 'APR_PROFIT': 3800, 'MAY_REVENUE': 14000, 'MAY_PROFIT': 4000, 'JUN_REVENUE': 15000, 'JUN_PROFIT': 4200, 'JUL_REVENUE': 16000, 'JUL_PROFIT': 4500, 'AUG_REVENUE': 17000, 'AUG_PROFIT': 4800, 'SEP_REVENUE': 18000, 'SEP_PROFIT': 5000, 'OCT_REVENUE': 19000, 'OCT_PROFIT': 5200, 'NOV_REVENUE': 20000, 'NOV_PROFIT': 5500, 'DEC_REVENUE': 21000, 'DEC_PROFIT': 5800}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'HR', 'JAN_REVENUE': 10000, 'JAN_PROFIT': 3000, 'FEB_REVENUE': 11000, 'FEB_PROFIT': 3200, 'MAR_REVENUE': 12000, 'MAR_PROFIT': 3500, 'APR_REVENUE': 13000, 'APR_PROFIT': 3800, 'MAY_REVENUE': 14000, 'MAY_PROFIT': 4000, 'JUN_REVENUE': 15000, 'JUN_PROFIT': 4200, 'JUL_REVENUE': 16000, 'JUL_PROFIT': 4500, 'AUG_REVENUE': 17000, 'AUG_PROFIT': 4800, 'SEP_REVENUE': 18000, 'SEP_PROFIT': 5000, 'OCT_REVENUE': 19000, 'OCT_PROFIT': 5200, 'NOV_REVENUE': 20000, 'NOV_PROFIT': 5500, 'DEC_REVENUE': 21000, 'DEC_PROFIT': 5800}]
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index('DEPT_NAME').T.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Monthly Revenues and Profits for the HR Department Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 598 tokens
            "question": "What are the quarterly expenses and profit margins for the R&D department over the past year?",
            "raw_data": "[{'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15}, {'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15}, {'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [
    {'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15},
    {'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15},
    {'DEPT_NAME': 'R&D', 'Q1_EXPENSES': 15000, 'Q1_PROFIT_MARGIN': 0.10, 'Q2_EXPENSES': 16000, 'Q2_PROFIT_MARGIN': 0.12, 'Q3_EXPENSES': 17000, 'Q3_PROFIT_MARGIN': 0.14, 'Q4_EXPENSES': 18000, 'Q4_PROFIT_MARGIN': 0.15}
]
df = pd.DataFrame(data)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Quarter')
ax1.set_ylabel('Expenses', color=color)
ax1.plot(['Q1', 'Q2', 'Q3', 'Q4'], [15000, 16000, 17000, 18000], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Profit Margin', color=color)
ax2.plot(['Q1', 'Q2', 'Q3', 'Q4'], [0.10, 0.12, 0.14, 0.15], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Quarterly Expenses and Profit Margins for the R&D Department Over the Past Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 517 tokens
            "question": "What are the monthly sales and expenses trends for the Finance department over the last year?",
            "raw_data": "[{'DEPT_NAME': 'Finance', 'JAN_SALES': 25000, 'JAN_EXPENSES': 15000, 'FEB_SALES': 26000, 'FEB_EXPENSES': 16000, 'MAR_SALES': 27000, 'MAR_EXPENSES': 17000, 'APR_SALES': 28000, 'APR_EXPENSES': 18000, 'MAY_SALES': 29000, 'MAY_EXPENSES': 19000, 'JUN_SALES': 30000, 'JUN_EXPENSES': 20000, 'JUL_SALES': 31000, 'JUL_EXPENSES': 21000, 'AUG_SALES': 32000, 'AUG_EXPENSES': 22000, 'SEP_SALES': 33000, 'SEP_EXPENSES': 23000, 'OCT_SALES': 34000, 'OCT_EXPENSES': 24000, 'NOV_SALES': 35000, 'NOV_EXPENSES': 25000, 'DEC_SALES': 36000, 'DEC_EXPENSES': 26000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Finance', 'JAN_SALES': 25000, 'JAN_EXPENSES': 15000, 'FEB_SALES': 26000, 'FEB_EXPENSES': 16000, 'MAR_SALES': 27000, 'MAR_EXPENSES': 17000, 'APR_SALES': 28000, 'APR_EXPENSES': 18000, 'MAY_SALES': 29000, 'MAY_EXPENSES': 19000, 'JUN_SALES': 30000, 'JUN_EXPENSES': 20000, 'JUL_SALES': 31000, 'JUL_EXPENSES': 21000, 'AUG_SALES': 32000, 'AUG_EXPENSES': 22000, 'SEP_SALES': 33000, 'SEP_EXPENSES': 23000, 'OCT_SALES': 34000, 'OCT_EXPENSES': 24000, 'NOV_SALES': 35000, 'NOV_EXPENSES': 25000, 'DEC_SALES': 36000, 'DEC_EXPENSES': 26000}]
df = pd.DataFrame(data)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales', color=color)
ax1.plot(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 
         [25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Expenses', color=color)
ax2.plot(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 
         [15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Monthly Sales and Expenses Trends for the Finance Department Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        },
        {
            # 529 tokens
            "question": "What are the monthly profit and expenses trends for the Marketing department over the last year?",
            "raw_data": "[{'DEPT_NAME': 'Marketing', 'JAN_PROFIT': 22000, 'JAN_EXPENSES': 12000, 'FEB_PROFIT': 23000, 'FEB_EXPENSES': 13000, 'MAR_PROFIT': 24000, 'MAR_EXPENSES': 14000, 'APR_PROFIT': 25000, 'APR_EXPENSES': 15000, 'MAY_PROFIT': 26000, 'MAY_EXPENSES': 16000, 'JUN_PROFIT': 27000, 'JUN_EXPENSES': 17000, 'JUL_PROFIT': 28000, 'JUL_EXPENSES': 18000, 'AUG_PROFIT': 29000, 'AUG_EXPENSES': 19000, 'SEP_PROFIT': 30000, 'SEP_EXPENSES': 20000, 'OCT_PROFIT': 31000, 'OCT_EXPENSES': 21000, 'NOV_PROFIT': 32000, 'NOV_EXPENSES': 22000, 'DEC_PROFIT': 33000, 'DEC_EXPENSES': 23000}]",
            "expected_code": """
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = [{'DEPT_NAME': 'Marketing', 'JAN_PROFIT': 22000, 'JAN_EXPENSES': 12000, 'FEB_PROFIT': 23000, 'FEB_EXPENSES': 13000, 'MAR_PROFIT': 24000, 'MAR_EXPENSES': 14000, 'APR_PROFIT': 25000, 'APR_EXPENSES': 15000, 'MAY_PROFIT': 26000, 'MAY_EXPENSES': 16000, 'JUN_PROFIT': 27000, 'JUN_EXPENSES': 17000, 'JUL_PROFIT': 28000, 'JUL_EXPENSES': 18000, 'AUG_PROFIT': 29000, 'AUG_EXPENSES': 19000, 'SEP_PROFIT': 30000, 'SEP_EXPENSES': 20000, 'OCT_PROFIT': 31000, 'OCT_EXPENSES': 21000, 'NOV_PROFIT': 32000, 'NOV_EXPENSES': 22000, 'DEC_PROFIT': 33000, 'DEC_EXPENSES': 23000}]
df = pd.DataFrame(data)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Month')
ax1.set_ylabel('Profit', color=color)
ax1.plot(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 
         [22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Expenses', color=color)
ax2.plot(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 
         [12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Monthly Profit and Expenses Trends for the Marketing Department Over the Last Year')
plt.savefig('ground_truth_plot.png')
plt.close()
"""
        }
    ]
    return benchmark

def crop_image(image):
    """Crop the image to remove whitespace and focus on the plot area."""
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def normalize_image(image):
    """Normalize the image size and convert to monochrome (black and white)."""
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to a fixed size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

    # # Apply a threshold to convert grayscale to black and white
    threshold = 0.9
    image_array = (image_array > threshold).astype(np.float32)
    
    return image_array

def save_image(array, path):
    """Save a numpy array as an image."""
    image = Image.fromarray((array * 255).astype('uint8'))  # Convert binary array to image
    image.save(path)

def compare_images(img1_path, img2_path):
    """Compares two images, and returns structural similarity index measure and pixel differences"""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Crop the images to focus on the plot area
    img1 = crop_image(img1)
    img2 = crop_image(img2)

    # Normalize the images
    img1_array = normalize_image(img1)
    img2_array = normalize_image(img2)

    # Save normalized images for inspection
    save_image(img1_array, 'normalized_img1.png')
    save_image(img2_array, 'normalized_img2.png')

    # Calculate SSIM
    ssim_index, diff = ssim(img1_array, img2_array, full=True, data_range=1.0)
    diff = (diff * 255).astype("uint8")

    # Pixel-by-pixel comparison
    # pixel_diff = np.sum(diff > 0) / diff.size  # Percentage of different pixels

    print(f"SSIM: {ssim_index}")
    # print(f"Percentage of different pixels: {pixel_diff * 100}%")

    return ssim_index

def evaluate_text_to_visualization(llm, benchmark_data, difficulty):
    """Evaluate the Text-to-Visualization LLM using benchmark data."""
    executable_counter = 0  # Counter for executable generated codes
    bleu_scores = [] # Container for all bleu scores
    ssim_indexes = [] # Container for all ssim index scores
    query_times = [] # Container for all the query timings
    total_queries = len(benchmark_data)  # Total number of benchmark queries
    
    # Difficulty and Query Clarification
    print(f"Benchmarking {total_queries} number of queries for {difficulty} difficulty for Text-to-Python QLLM")

    for data in benchmark_data:
        question = data['question']
        raw_data = eval(data['raw_data'])  # Convert string representation of list to actual list
        expected_code = data['expected_code']
        
        # Default variable reset
        executable = False
        temp_bleu_score = 0
        temp_ssim_index = 0

        # Clean up plot files
        if os.path.exists('ground_truth_plot.png'):
            os.remove('ground_truth_plot.png')
        if os.path.exists('generated_plot.png'):
            os.remove('generated_plot.png')
        if os.path.exists('normalized_img1.png'):
            os.remove('normalized_img1.png')
        if os.path.exists('normalized_img2.png'):
            os.remove('normalized_img2.png')

        start_time = time.time()
        print(f"\n\nQuestion: {question}")

        # Ensure the ground truth code runs
        try:
            exec(expected_code)
            print("Ground truth code executed successfully.")
        except Exception as e:
            print(f"Error in executing ground truth code: {e}")
            continue
        
        try:
            # Generate the plot code
            generated_code = LLMConfiguration.generate_plot_code(llm, question, raw_data)
            if generated_code != -1:
                # Modify the generated code to save the plot to 'generated_plot.png'
                generated_code += "\nplt.savefig('generated_plot.png')\nplt.close()"

                # Time logging
                end_time = time.time()
                query_time = end_time - start_time
                query_times.append(query_time)

                try:
                    # Execute and validate the generated code
                    exec(generated_code)
                    print("Generated code executed successfully.")
                    executable = True
                    executable_counter += 1  # Increment executable counter if code runs successfully
                except Exception as e:
                    print(f"Error in executing generated python code: {e}")

                # Evaluate the code using BLEU for similarity
                reference_tokens = [nltk.word_tokenize(expected_code)]
                candidate_tokens = nltk.word_tokenize(generated_code)
                smoothing_function = SmoothingFunction().method4
                temp_bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
                print(f"BLEU Score: {temp_bleu_score}")
                # Add current bleu score to container
                bleu_scores.append(temp_bleu_score)

                # Compare the generated plot with the ground truth plot
                if os.path.exists('generated_plot.png'):
                    temp_ssim_index = compare_images('ground_truth_plot.png', 'generated_plot.png')
                    ssim_indexes.append(temp_ssim_index)
                    print(f"SSIM Index: {temp_ssim_index}")
                else:
                    print("Generated plot not created, skipping image comparison.")
            else:
                print("Test case exceeded defined context length context for Text-to-Python QLLM. Skipping... ")

        except Exception as e:
            print(f"Error with python visualisation generation: {e}")
    
        # Save results to database
        DatabaseConfiguration.store_vis_stat(question, temp_bleu_score, temp_ssim_index, executable, difficulty)
        print("Added Visualization Stat to DB")

    # Calculate a summary of metrics for visualization difficulty
    executable_rate = executable_counter / total_queries
    average_bleu_score = sum(bleu_scores) / total_queries
    average_ssim_index = sum(ssim_indexes) / executable_counter
    average_latency = sum(query_times) / total_queries

    print(f"Executable Rate: {executable_rate * 100}%")
    print(f"Average BLEU Score: {average_bleu_score:.2f}")
    print(f"Average SSIM Score: {average_ssim_index:.2f}")
    print(f"Average Latency: {average_latency:.2f} seconds")

    # Store result summary to database
    DatabaseConfiguration.store_vis_summary(difficulty, average_bleu_score, average_ssim_index, average_latency, executable_rate)
    print("Added VIS summary stat to vis_summary_stats")

def run_visualization_benchmark():
    python_llm = LLMConfiguration.deploy_python_llama()

    easy_benchmark_data = load_easy_benchmark()
    medium_benchmark_data = load_medium_benchmark()
    hard_benchmark_data = load_hard_benchmark()

    evaluate_text_to_visualization(python_llm, easy_benchmark_data, "easy")
    evaluate_text_to_visualization(python_llm, medium_benchmark_data, "medium")
    evaluate_text_to_visualization(python_llm, hard_benchmark_data, "hard")