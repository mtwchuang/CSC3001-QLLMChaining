# Local libraries
import LLMConfiguration
import DatabaseConfiguration
# Others
from rouge import Rouge
from bert_score import score
import numpy as np
import time

# 15 easy test cases
def load_easy_benchmark():
    """Loads benchmark dataset of text context 100-250 tokens to summarize"""
    benchmark_data = [
        {
            # 146
            "question": "What are the names of the employees in the Sales department?",
            "raw_data":"[{'EMPLOYEE_NAME': 'John Doe', 'DEPT_NAME': 'Sales'}, {'EMPLOYEE_NAME': 'Jane Smith', 'DEPT_NAME': 'Sales'}]",
            "summary": "Employees in the Sales department:\n- John Doe\n- Jane Smith"
        },
        {
            # 145
            "question": "What are the total sales for each product?",
            "raw_data":"[{'PRODUCT_NAME': 'Product A', 'TOTAL_SALES': 1000}, {'PRODUCT_NAME': 'Product B', 'TOTAL_SALES': 1500}]",
            "summary": "Total sales for each product:\n- Product A: 1000\n- Product B: 1500"
        },
        {
            # 140
            "question": "List all vendor names from the USA.",
            "raw_data":"[{'VENDOR_NAME': 'Vendor X', 'COUNTRY_CODE': 'USA'}, {'VENDOR_NAME': 'Vendor Y', 'COUNTRY_CODE': 'USA'}]",
            "summary": "Vendors from the USA:\n- Vendor X\n- Vendor Y"
        },
        {
            # 150
            "question": "What are the names of products with sales greater than 500?",
            "raw_data":"[{'PRODUCT_NAME': 'Product A', 'TOTAL_SALES': 1000}, {'PRODUCT_NAME': 'Product B', 'TOTAL_SALES': 700}]",
            "summary": "Products with sales greater than 500:\n- Product A: 1000\n- Product B: 700"
        },
        {
            # 142
            "question": "List all departments and their total number of employees.",
            "raw_data":"[{'DEPT_NAME': 'Sales', 'EMPLOYEE_COUNT': 10}, {'DEPT_NAME': 'Marketing', 'EMPLOYEE_COUNT': 8}]",
            "summary": "Departments and their total number of employees:\n- Sales: 10\n- Marketing: 8"
        },
        {
            # 200
            "question": "What are the names of employees who have worked on projects with payment amounts exceeding 10,000?",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'PAYMENT_AMOUNT': 12000}, {'EMPLOYEE_NAME': 'Jane Smith', 'PAYMENT_AMOUNT': 15000}, {'EMPLOYEE_NAME': 'Alice Johnson', 'PAYMENT_AMOUNT': 18000}]",
            "summary": "Employees who have worked on projects with payment amounts exceeding 10,000:\n- John Doe: 12,000\n- Jane Smith: 15,000\n- Alice Johnson: 18,000"
        },
        {
            # 175
            "question": "List the names and sales amounts of products sold in the last quarter.",
            "raw_data": "[{'PRODUCT_NAME': 'Product X', 'SALES_AMOUNT': 2000}, {'PRODUCT_NAME': 'Product Y', 'SALES_AMOUNT': 3000}, {'PRODUCT_NAME': 'Product Z', 'SALES_AMOUNT': 4000}]",
            "summary": "Products sold in the last quarter:\n- Product X: 2,000\n- Product Y: 3,000\n- Product Z: 4,000"
        },
        {
            # 170
            "question": "What are the total sales for each department?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'TOTAL_SALES': 50000}, {'DEPT_NAME': 'Marketing', 'TOTAL_SALES': 60000}, {'DEPT_NAME': 'Engineering', 'TOTAL_SALES': 70000}]",
            "summary": "Total sales for each department:\n- Sales: 50,000\n- Marketing: 60,000\n- Engineering: 70,000"
        },
        {
            # 187
            "question": "List the employees who have received bonuses and their respective bonus amounts.",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'BONUS_AMOUNT': 1000}, {'EMPLOYEE_NAME': 'Jane Smith', 'BONUS_AMOUNT': 1500}, {'EMPLOYEE_NAME': 'Alice Johnson', 'BONUS_AMOUNT': 2000}]",
            "summary": "Employees who have received bonuses:\n- John Doe: 1,000\n- Jane Smith: 1,500\n- Alice Johnson: 2,000"
        },
        {
            # 186
            "question": "Show the vendor names and their corresponding payment amounts for the last fiscal year.",
            "raw_data": "[{'VENDOR_NAME': 'Vendor A', 'PAYMENT_AMOUNT': 10000}, {'VENDOR_NAME': 'Vendor B', 'PAYMENT_AMOUNT': 20000}, {'VENDOR_NAME': 'Vendor C', 'PAYMENT_AMOUNT': 30000}]",
            "summary": "Vendors and their payment amounts for the last fiscal year:\n- Vendor A: 10,000\n- Vendor B: 20,000\n- Vendor C: 30,000"
        },
        {
            # 158
            "question": "List all departments along with the number of projects they have handled.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'PROJECT_COUNT': 5}, {'DEPT_NAME': 'Marketing', 'PROJECT_COUNT': 8}, {'DEPT_NAME': 'Engineering', 'PROJECT_COUNT': 12}]",
            "summary": "Departments and the number of projects handled:\n- Sales: 5\n- Marketing: 8\n- Engineering: 12"
        },
        {
            # 185
            "question": "What are the names of vendors with payment amounts greater than the average?",
            "raw_data": "[{'VENDOR_NAME': 'Vendor X', 'PAYMENT_AMOUNT': 25000}, {'VENDOR_NAME': 'Vendor Y', 'PAYMENT_AMOUNT': 30000}, {'VENDOR_NAME': 'Vendor Z', 'PAYMENT_AMOUNT': 35000}]",
            "summary": "Vendors with payment amounts greater than the average:\n- Vendor X: 25,000\n- Vendor Y: 30,000\n- Vendor Z: 35,000"
        },
        {
            # 184
            "question": "List the products and their sales amounts that contribute to over 50% of the total sales.",
            "raw_data": "[{'PRODUCT_NAME': 'Product A', 'SALES_AMOUNT': 50000}, {'PRODUCT_NAME': 'Product B', 'SALES_AMOUNT': 60000}, {'PRODUCT_NAME': 'Product C', 'SALES_AMOUNT': 70000}]",
            "summary": "Products contributing over 50% of total sales:\n- Product A: 50,000\n- Product B: 60,000\n- Product C: 70,000"
        },
        {
            # 174
            "question": "Show the employee names and the total number of projects they have worked on in the past year.",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'PROJECT_COUNT': 4}, {'EMPLOYEE_NAME': 'Jane Smith', 'PROJECT_COUNT': 5}, {'EMPLOYEE_NAME': 'Alice Johnson', 'PROJECT_COUNT': 6}]",
            "summary": "Employees and the total number of projects they have worked on in the past year:\n- John Doe: 4 projects\n- Jane Smith: 5 projects\n- Alice Johnson: 6 projects"
        },
        {
            # 223 tokens
            "question": "What are the sales figures for each product category in the last quarter?",
            "raw_data": "[{'CATEGORY_NAME': 'Electronics', 'SALES': 100000}, {'CATEGORY_NAME': 'Furniture', 'SALES': 150000}, {'CATEGORY_NAME': 'Clothing', 'SALES': 120000}, {'CATEGORY_NAME': 'Sports', 'SALES': 110000}, {'CATEGORY_NAME': 'Toys', 'SALES': 130000}]",
            "summary": "Sales figures for each product category in the last quarter:\n- Electronics: 100,000\n- Furniture: 150,000\n- Clothing: 120,000\n- Sports: 110,000\n- Toys: 130,000"
        },
        # # 216 tokens
        # {
        #     "question": "List the names of employees and the number of projects they have completed this year.",
        #     "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'PROJECT_COUNT': 5}, {'EMPLOYEE_NAME': 'Jane Smith', 'PROJECT_COUNT': 6}, {'EMPLOYEE_NAME': 'Alice Johnson', 'PROJECT_COUNT': 7}, {'EMPLOYEE_NAME': 'Bob Brown', 'PROJECT_COUNT': 8}, {'EMPLOYEE_NAME': 'Charlie Black', 'PROJECT_COUNT': 9}]",
        #     "summary": "Employees and the number of projects they have completed this year:\n- John Doe: 5 projects\n- Jane Smith: 6 projects\n- Alice Johnson: 7 projects\n- Bob Brown: 8 projects\n- Charlie Black: 9 projects"
        # },
        # # 211 tokens
        # {
        #     "question": "What are the total sales and number of transactions for each region?",
        #     "raw_data": "[{'REGION_NAME': 'North', 'TOTAL_SALES': 80000, 'TRANSACTION_COUNT': 120}, {'REGION_NAME': 'South', 'TOTAL_SALES': 90000, 'TRANSACTION_COUNT': 130}, {'REGION_NAME': 'East', 'TOTAL_SALES': 100000, 'TRANSACTION_COUNT': 140}]",
        #     "summary": "Total sales and number of transactions for each region:\n- North: 80,000 (120 transactions)\n- South: 90,000 (130 transactions)\n- East: 100,000 (140 transactions)"
        # }
    ]
    return benchmark_data

# 15 easy test cases
def load_medium_benchmark():
    """Loads benchmark dataset of text context 250 - 500 tokens tokens to summarize"""
    benchmark_data = [
        {
            # 301 tokens
            "question": "List the top 5 employees with the highest sales and their respective departments.",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 200000}, {'EMPLOYEE_NAME': 'Jane Smith', 'DEPT_NAME': 'Marketing', 'TOTAL_SALES': 180000}, {'EMPLOYEE_NAME': 'Alice Johnson', 'DEPT_NAME': 'Engineering', 'TOTAL_SALES': 170000}, {'EMPLOYEE_NAME': 'Bob Brown', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 160000}, {'EMPLOYEE_NAME': 'Charlie Black', 'DEPT_NAME': 'Sales', 'TOTAL_SALES': 150000}]",
            "summary": "Top 5 employees with the highest sales and their respective departments:\n- John Doe (Sales): 200,000\n- Jane Smith (Marketing): 180,000\n- Alice Johnson (Engineering): 170,000\n- Bob Brown (Sales): 160,000\n- Charlie Black (Sales): 150,000"
        },
        {
            # 464 tokens
            "question": "What are the monthly sales for each product in the last 6 months?",
            "raw_data": "[{'PRODUCT_NAME': 'Product A', 'MONTH': 'January', 'SALES': 10000}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'February', 'SALES': 12000}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'March', 'SALES': 13000}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'April', 'SALES': 14000}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'May', 'SALES': 15000}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'June', 'SALES': 16000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'January', 'SALES': 20000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'February', 'SALES': 22000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'March', 'SALES': 23000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'April', 'SALES': 24000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'May', 'SALES': 25000}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'June', 'SALES': 26000}]",
            "summary": "Monthly sales for each product in the last 6 months:\n- Product A:\n  - January: 10,000\n  - February: 12,000\n  - March: 13,000\n  - April: 14,000\n  - May: 15,000\n  - June: 16,000\n- Product B:\n  - January: 20,000\n  - February: 22,000\n  - March: 23,000\n  - April: 24,000\n  - May: 25,000\n  - June: 26,000"
        },
        {
            # 334 tokens
            "question": "What are the total expenses and revenues for each department in the last fiscal year?",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'TOTAL_EXPENSES': 30000, 'TOTAL_REVENUES': 500000}, {'DEPT_NAME': 'Marketing', 'TOTAL_EXPENSES': 20000, 'TOTAL_REVENUES': 400000}, {'DEPT_NAME': 'Engineering', 'TOTAL_EXPENSES': 25000, 'TOTAL_REVENUES': 450000}, {'DEPT_NAME': 'HR', 'TOTAL_EXPENSES': 15000, 'TOTAL_REVENUES': 350000}, {'DEPT_NAME': 'Finance', 'TOTAL_EXPENSES': 10000, 'TOTAL_REVENUES': 300000}]",
            "summary": "Total expenses and revenues for each department in the last fiscal year:\n- Sales: 30,000 expenses, 500,000 revenues\n- Marketing: 20,000 expenses, 400,000 revenues\n- Engineering: 25,000 expenses, 450,000 revenues\n- HR: 15,000 expenses, 350,000 revenues\n- Finance: 10,000 expenses, 300,000 revenues"
        },
        {
            # 471 tokens
            "question": "What are the monthly sales figures and the number of returns for each product?",
            "raw_data": "[{'PRODUCT_NAME': 'Product A', 'MONTH': 'January', 'SALES': 10000, 'RETURNS': 50}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'February', 'SALES': 12000, 'RETURNS': 60}, {'PRODUCT_NAME': 'Product A', 'MONTH': 'March', 'SALES': 13000, 'RETURNS': 70}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'January', 'SALES': 20000, 'RETURNS': 100}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'February', 'SALES': 22000, 'RETURNS': 110}, {'PRODUCT_NAME': 'Product B', 'MONTH': 'March', 'SALES': 23000, 'RETURNS': 120}, {'PRODUCT_NAME': 'Product C', 'MONTH': 'January', 'SALES': 15000, 'RETURNS': 75}, {'PRODUCT_NAME': 'Product C', 'MONTH': 'February', 'SALES': 17000, 'RETURNS': 85}, {'PRODUCT_NAME': 'Product C', 'MONTH': 'March', 'SALES': 18000, 'RETURNS': 95}]",
            "summary": "Monthly sales figures and number of returns for each product:\n- Product A:\n  - January: 10,000 sales, 50 returns\n  - February: 12,000 sales, 60 returns\n  - March: 13,000 sales, 70 returns\n- Product B:\n  - January: 20,000 sales, 100 returns\n  - February: 22,000 sales, 110 returns\n  - March: 23,000 sales, 120 returns\n- Product C:\n  - January: 15,000 sales, 75 returns\n  - February: 17,000 sales, 85 returns\n  - March: 18,000 sales, 95 returns"
        },
        {
            # 302 tokens
            "question": "List the top 5 vendors with the highest total sales and the number of products they have sold.",
            "raw_data": "[{'VENDOR_NAME': 'Vendor X', 'TOTAL_SALES': 300000, 'PRODUCT_COUNT': 50}, {'VENDOR_NAME': 'Vendor Y', 'TOTAL_SALES': 280000, 'PRODUCT_COUNT': 45}, {'VENDOR_NAME': 'Vendor Z', 'TOTAL_SALES': 260000, 'PRODUCT_COUNT': 40}, {'VENDOR_NAME': 'Vendor A', 'TOTAL_SALES': 250000, 'PRODUCT_COUNT': 35}, {'VENDOR_NAME': 'Vendor B', 'TOTAL_SALES': 240000, 'PRODUCT_COUNT': 30}]",
            "summary": "Top 5 vendors with the highest total sales and the number of products they have sold:\n- Vendor X: 300,000 total sales, 50 products\n- Vendor Y: 280,000 total sales, 45 products\n- Vendor Z: 260,000 total sales, 40 products\n- Vendor A: 250,000 total sales, 35 products\n- Vendor B: 240,000 total sales, 30 products"
        },
        {
            # 432 tokens
            "question": "What are the annual performance metrics for each employee in terms of sales, expenses, and profit?",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'ANNUAL_SALES': 300000, 'ANNUAL_EXPENSES': 200000, 'ANNUAL_PROFIT': 100000}, {'EMPLOYEE_NAME': 'Jane Smith', 'ANNUAL_SALES': 250000, 'ANNUAL_EXPENSES': 150000, 'ANNUAL_PROFIT': 100000}, {'EMPLOYEE_NAME': 'Alice Johnson', 'ANNUAL_SALES': 200000, 'ANNUAL_EXPENSES': 100000, 'ANNUAL_PROFIT': 100000}, {'EMPLOYEE_NAME': 'Bob Brown', 'ANNUAL_SALES': 150000, 'ANNUAL_EXPENSES': 50000, 'ANNUAL_PROFIT': 100000}, {'EMPLOYEE_NAME': 'Charlie Black', 'ANNUAL_SALES': 100000, 'ANNUAL_EXPENSES': 50000, 'ANNUAL_PROFIT': 50000}]",
            "summary": "Annual performance metrics for each employee:\n- John Doe: 300,000 sales, 200,000 expenses, 100,000 profit\n- Jane Smith: 250,000 sales, 150,000 expenses, 100,000 profit\n- Alice Johnson: 200,000 sales, 100,000 expenses, 100,000 profit\n- Bob Brown: 150,000 sales, 50,000 expenses, 100,000 profit\n- Charlie Black: 100,000 sales, 50,000 expenses, 50,000 profit"
        },
        {
            # 297 tokens
            "question": "Provide a summary of the total revenues and expenses for the company's five departments.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'REVENUES': 500000, 'EXPENSES': 30000}, {'DEPT_NAME': 'Marketing', 'REVENUES': 400000, 'EXPENSES': 20000}, {'DEPT_NAME': 'Engineering', 'REVENUES': 450000, 'EXPENSES': 25000}, {'DEPT_NAME': 'HR', 'REVENUES': 350000, 'EXPENSES': 15000}, {'DEPT_NAME': 'Finance', 'REVENUES': 300000, 'EXPENSES': 10000}]",
            "summary": "Total revenues and expenses for the company's departments:\n- Sales: 500,000 revenues, 30,000 expenses\n- Marketing: 400,000 revenues, 20,000 expenses\n- Engineering: 450,000 revenues, 25,000 expenses\n- HR: 350,000 revenues, 15,000 expenses\n- Finance: 300,000 revenues, 10,000 expenses"
        },
        {
            # 321 tokens
            "question": "What are the total expenditures and the average transaction amount for each vendor?",
            "raw_data": "[{'VENDOR_NAME': 'Vendor A', 'TOTAL_EXPENDITURES': 120000, 'TRANSACTION_COUNT': 240}, {'VENDOR_NAME': 'Vendor B', 'TOTAL_EXPENDITURES': 95000, 'TRANSACTION_COUNT': 190}, {'VENDOR_NAME': 'Vendor C', 'TOTAL_EXPENDITURES': 110000, 'TRANSACTION_COUNT': 220}, {'VENDOR_NAME': 'Vendor D', 'TOTAL_EXPENDITURES': 105000, 'TRANSACTION_COUNT': 210}, {'VENDOR_NAME': 'Vendor E', 'TOTAL_EXPENDITURES': 98000, 'TRANSACTION_COUNT': 196}]",
            "summary": "Total expenditures and average transaction amount for each vendor:\n- Vendor A: 120,000 expenditures (240 transactions)\n- Vendor B: 95,000 expenditures (190 transactions)\n- Vendor C: 110,000 expenditures (220 transactions)\n- Vendor D: 105,000 expenditures (210 transactions)\n- Vendor E: 98,000 expenditures (196 transactions)"
        },
        {
            # 372 tokens
            "question": "Summarize the quarterly revenue and expense data for each department.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'QUARTER': 'Q1', 'REVENUE': 150000, 'EXPENSE': 7500}, {'DEPT_NAME': 'Sales', 'QUARTER': 'Q2', 'REVENUE': 130000, 'EXPENSE': 6500}, {'DEPT_NAME': 'Marketing', 'QUARTER': 'Q1', 'REVENUE': 120000, 'EXPENSE': 7000}, {'DEPT_NAME': 'Marketing', 'QUARTER': 'Q2', 'REVENUE': 110000, 'EXPENSE': 6000}, {'DEPT_NAME': 'Engineering', 'QUARTER': 'Q1', 'REVENUE': 135000, 'EXPENSE': 8500}, {'DEPT_NAME': 'Engineering', 'QUARTER': 'Q2', 'REVENUE': 140000, 'EXPENSE': 9000}]",
            "summary": "Quarterly revenue and expense data for each department:\n- Sales: Q1: 150,000 revenue, 7,500 expense; Q2: 130,000 revenue, 6,500 expense\n- Marketing: Q1: 120,000 revenue, 7,000 expense; Q2: 110,000 revenue, 6,000 expense\n- Engineering: Q1: 135,000 revenue, 8,500 expense; Q2: 140,000 revenue, 9,000 expense"
        },
        {
            # 309 tokens
            "question": "What are the total payments and the number of transactions for each vendor in the last quarter?",
            "raw_data": "[{'VENDOR_NAME': 'Vendor X', 'TOTAL_PAYMENTS': 100000, 'TRANSACTIONS': 200}, {'VENDOR_NAME': 'Vendor Y', 'TOTAL_PAYMENTS': 95000, 'TRANSACTIONS': 180}, {'VENDOR_NAME': 'Vendor Z', 'TOTAL_PAYMENTS': 120000, 'TRANSACTIONS': 210}, {'VENDOR_NAME': 'Vendor W', 'TOTAL_PAYMENTS': 110000, 'TRANSACTIONS': 190}, {'VENDOR_NAME': 'Vendor V', 'TOTAL_PAYMENTS': 105000, 'TRANSACTIONS': 205}]",
            "summary": "Total payments and number of transactions for each vendor in the last quarter:\n- Vendor X: 100,000 payments (200 transactions)\n- Vendor Y: 95,000 payments (180 transactions)\n- Vendor Z: 120,000 payments (210 transactions)\n- Vendor W: 110,000 payments (190 transactions)\n- Vendor V: 105,000 payments (205 transactions)"
        },
        {
            # 309 tokens
            "question": "Summarize the sales and number of units sold for each product category.",
            "raw_data": "[{'CATEGORY': 'Electronics', 'TOTAL_SALES': 200000, 'UNITS_SOLD': 4000}, {'CATEGORY': 'Furniture', 'TOTAL_SALES': 150000, 'UNITS_SOLD': 3000}, {'CATEGORY': 'Clothing', 'TOTAL_SALES': 180000, 'UNITS_SOLD': 3600}, {'CATEGORY': 'Toys', 'TOTAL_SALES': 140000, 'UNITS_SOLD': 2800}, {'CATEGORY': 'Books', 'TOTAL_SALES': 120000, 'UNITS_SOLD': 2400}]",
            "summary": "Sales and number of units sold for each product category:\n- Electronics: 200,000 sales (4000 units)\n- Furniture: 150,000 sales (3000 units)\n- Clothing: 180,000 sales (3600 units)\n- Toys: 140,000 sales (2800 units)\n- Books: 120,000 sales (2400 units)"
        },
        {
            # 295 tokens
            "question": "Provide a detailed report of the number of employees and the total hours worked in each department.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'EMPLOYEE_COUNT': 50, 'TOTAL_HOURS': 8000}, {'DEPT_NAME': 'Marketing', 'EMPLOYEE_COUNT': 40, 'TOTAL_HOURS': 6400}, {'DEPT_NAME': 'Engineering', 'EMPLOYEE_COUNT': 60, 'TOTAL_HOURS': 9600}, {'DEPT_NAME': 'HR', 'EMPLOYEE_COUNT': 30, 'TOTAL_HOURS': 4800}, {'DEPT_NAME': 'Finance', 'EMPLOYEE_COUNT': 35, 'TOTAL_HOURS': 5600}]",
            "summary": "Number of employees and total hours worked in each department:\n- Sales: 50 employees, 8000 hours\n- Marketing: 40 employees, 6400 hours\n- Engineering: 60 employees, 9600 hours\n- HR: 30 employees, 4800 hours\n- Finance: 35 employees, 5600 hours"
        },
        {
            # 440 tokens
            "question": "What are the total sales and number of units sold for each product in the last year?",
            "raw_data": "[{'PRODUCT_NAME': 'Product A', 'TOTAL_SALES': 120000, 'UNITS_SOLD': 4000}, {'PRODUCT_NAME': 'Product B', 'TOTAL_SALES': 110000, 'UNITS_SOLD': 3500}, {'PRODUCT_NAME': 'Product C', 'TOTAL_SALES': 130000, 'UNITS_SOLD': 4200}, {'PRODUCT_NAME': 'Product D', 'TOTAL_SALES': 115000, 'UNITS_SOLD': 3800}, {'PRODUCT_NAME': 'Product E', 'TOTAL_SALES': 105000, 'UNITS_SOLD': 3400}, {'PRODUCT_NAME': 'Product F', 'TOTAL_SALES': 125000, 'UNITS_SOLD': 4100}, {'PRODUCT_NAME': 'Product G', 'TOTAL_SALES': 135000, 'UNITS_SOLD': 4300}, {'PRODUCT_NAME': 'Product H', 'TOTAL_SALES': 140000, 'UNITS_SOLD': 4500}]",
            "summary": "Total sales and number of units sold for each product in the last year:\n- Product A: 120,000 sales (4,000 units)\n- Product B: 110,000 sales (3,500 units)\n- Product C: 130,000 sales (4,200 units)\n- Product D: 115,000 sales (3,800 units)\n- Product E: 105,000 sales (3,400 units)\n- Product F: 125,000 sales (4,100 units)\n- Product G: 135,000 sales (4,300 units)\n- Product H: 140,000 sales (4,500 units)"
        },
        {
            # 338 tokens
            "question": "What are the annual sales and average sales per transaction for each product in the Electronics category?",
            "raw_data": "[{'PRODUCT_NAME': 'Laptop', 'ANNUAL_SALES': 120000, 'TRANSACTION_COUNT': 400}, {'PRODUCT_NAME': 'Smartphone', 'ANNUAL_SALES': 100000, 'TRANSACTION_COUNT': 500}, {'PRODUCT_NAME': 'Tablet', 'ANNUAL_SALES': 80000, 'TRANSACTION_COUNT': 300}, {'PRODUCT_NAME': 'Headphones', 'ANNUAL_SALES': 50000, 'TRANSACTION_COUNT': 600}, {'PRODUCT_NAME': 'Smartwatch', 'ANNUAL_SALES': 60000, 'TRANSACTION_COUNT': 400}, {'PRODUCT_NAME': 'Camera', 'ANNUAL_SALES': 90000, 'TRANSACTION_COUNT': 450}]",
            "summary": "Annual sales and average sales per transaction for each product in the Electronics category:\n- Laptop: 120,000 annual sales, 300 average sales per transaction\n- Smartphone: 100,000 annual sales, 200 average sales per transaction\n- Tablet: 80,000 annual sales, 267 average sales per transaction\n- Headphones: 50,000 annual sales, 83 average sales per transaction\n- Smartwatch: 60,000 annual sales, 150 average sales per transaction\n- Camera: 90,000 annual sales, 200 average sales per transaction"
        },
        {
            # 334 tokens
            "question": "Provide a detailed breakdown of the total hours worked and average hours worked per employee for each department.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'TOTAL_HOURS': 8000, 'EMPLOYEE_COUNT': 50}, {'DEPT_NAME': 'Marketing', 'TOTAL_HOURS': 6400, 'EMPLOYEE_COUNT': 40}, {'DEPT_NAME': 'Engineering', 'TOTAL_HOURS': 9600, 'EMPLOYEE_COUNT': 60}, {'DEPT_NAME': 'HR', 'TOTAL_HOURS': 4800, 'EMPLOYEE_COUNT': 30}, {'DEPT_NAME': 'Finance', 'TOTAL_HOURS': 5600, 'EMPLOYEE_COUNT': 35}, {'DEPT_NAME': 'Support', 'TOTAL_HOURS': 7200, 'EMPLOYEE_COUNT': 45}]",
            "summary": "Total hours worked and average hours worked per employee for each department:\n- Sales: 8000 total hours, 160 average hours per employee\n- Marketing: 6400 total hours, 160 average hours per employee\n- Engineering: 9600 total hours, 160 average hours per employee\n- HR: 4800 total hours, 160 average hours per employee\n- Finance: 5600 total hours, 160 average hours per employee\n- Support: 7200 total hours, 160 average hours per employee"
        }
    ]
    return benchmark_data

# 15 hard test cases
def load_hard_benchmark():
    """Loads benchmark dataset of text context > 500 tokens to summarize"""
    benchmark_data = [
        {
            # 600 tokens
            "question": "What are the quarterly sales and number of transactions for each product category?",
            "raw_data": "[{'CATEGORY_NAME': 'Electronics', 'Q1_SALES': 50000, 'Q1_TRANSACTIONS': 300, 'Q2_SALES': 60000, 'Q2_TRANSACTIONS': 350, 'Q3_SALES': 70000, 'Q3_TRANSACTIONS': 400, 'Q4_SALES': 80000, 'Q4_TRANSACTIONS': 450}, {'CATEGORY_NAME': 'Furniture', 'Q1_SALES': 40000, 'Q1_TRANSACTIONS': 250, 'Q2_SALES': 50000, 'Q2_TRANSACTIONS': 300, 'Q3_SALES': 60000, 'Q3_TRANSACTIONS': 350, 'Q4_SALES': 70000, 'Q4_TRANSACTIONS': 400}, {'CATEGORY_NAME': 'Clothing', 'Q1_SALES': 30000, 'Q1_TRANSACTIONS': 200, 'Q2_SALES': 40000, 'Q2_TRANSACTIONS': 250, 'Q3_SALES': 50000, 'Q3_TRANSACTIONS': 300, 'Q4_SALES': 60000, 'Q4_TRANSACTIONS': 350}, {'CATEGORY_NAME': 'Toys', 'Q1_SALES': 20000, 'Q1_TRANSACTIONS': 150, 'Q2_SALES': 30000, 'Q2_TRANSACTIONS': 200, 'Q3_SALES': 40000, 'Q3_TRANSACTIONS': 250, 'Q4_SALES': 50000, 'Q4_TRANSACTIONS': 300}]",
            "summary": "Quarterly sales and number of transactions for each product category:\n- Electronics:\n  - Q1: 50,000 sales, 300 transactions\n  - Q2: 60,000 sales, 350 transactions\n  - Q3: 70,000 sales, 400 transactions\n  - Q4: 80,000 sales, 450 transactions\n- Furniture:\n  - Q1: 40,000 sales, 250 transactions\n  - Q2: 50,000 sales, 300 transactions\n  - Q3: 60,000 sales, 350 transactions\n  - Q4: 70,000 sales, 400 transactions\n- Clothing:\n  - Q1: 30,000 sales, 200 transactions\n  - Q2: 40,000 sales, 250 transactions\n  - Q3: 50,000 sales, 300 transactions\n  - Q4: 60,000 sales, 350 transactions\n- Toys:\n  - Q1: 20,000 sales, 150 transactions\n  - Q2: 30,000 sales, 200 transactions\n  - Q3: 40,000 sales, 250 transactions\n  - Q4: 50,000 sales, 300 transactions"
        },
        {
            # 502 tokens
            "question": "List the performance metrics of each department for the last year.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'REVENUE': 500000, 'EXPENSES': 300000, 'PROFIT': 200000, 'EMPLOYEE_COUNT': 50, 'CUSTOMER_SATISFACTION': 85}, {'DEPT_NAME': 'Marketing', 'REVENUE': 400000, 'EXPENSES': 200000, 'PROFIT': 200000, 'EMPLOYEE_COUNT': 40, 'CUSTOMER_SATISFACTION': 80}, {'DEPT_NAME': 'Engineering', 'REVENUE': 450000, 'EXPENSES': 250000, 'PROFIT': 200000, 'EMPLOYEE_COUNT': 45, 'CUSTOMER_SATISFACTION': 90}, {'DEPT_NAME': 'HR', 'REVENUE': 350000, 'EXPENSES': 150000, 'PROFIT': 200000, 'EMPLOYEE_COUNT': 35, 'CUSTOMER_SATISFACTION': 75}, {'DEPT_NAME': 'Finance', 'REVENUE': 300000, 'EXPENSES': 100000, 'PROFIT': 200000, 'EMPLOYEE_COUNT': 30, 'CUSTOMER_SATISFACTION': 95}]",
            "summary": "Performance metrics of each department for the last year:\n- Sales:\n  - Revenue: 500,000\n  - Expenses: 300,000\n  - Profit: 200,000\n  - Employee Count: 50\n  - Customer Satisfaction: 85\n- Marketing:\n  - Revenue: 400,000\n  - Expenses: 200,000\n  - Profit: 200,000\n  - Employee Count: 40\n  - Customer Satisfaction: 80\n- Engineering:\n  - Revenue: 450,000\n  - Expenses: 250,000\n  - Profit: 200,000\n  - Employee Count: 45\n  - Customer Satisfaction: 90\n- HR:\n  - Revenue: 350,000\n  - Expenses: 150,000\n  - Profit: 200,000\n  - Employee Count: 35\n  - Customer Satisfaction: 75\n- Finance:\n  - Revenue: 300,000\n  - Expenses: 100,000\n  - Profit: 200,000\n  - Employee Count: 30\n  - Customer Satisfaction: 95"
        },
        {
            # 626 tokens
            "question": "List the monthly performance metrics for each department in terms of revenue, expenses, and profit.",
            "raw_data": "[{'DEPT_NAME': 'Sales', 'MONTH': 'January', 'REVENUE': 50000, 'EXPENSES': 30000, 'PROFIT': 20000}, {'DEPT_NAME': 'Sales', 'MONTH': 'February', 'REVENUE': 60000, 'EXPENSES': 35000, 'PROFIT': 25000}, {'DEPT_NAME': 'Sales', 'MONTH': 'March', 'REVENUE': 70000, 'EXPENSES': 40000, 'PROFIT': 30000}, {'DEPT_NAME': 'Marketing', 'MONTH': 'January', 'REVENUE': 40000, 'EXPENSES': 20000, 'PROFIT': 20000}, {'DEPT_NAME': 'Marketing', 'MONTH': 'February', 'REVENUE': 50000, 'EXPENSES': 25000, 'PROFIT': 25000}, {'DEPT_NAME': 'Marketing', 'MONTH': 'March', 'REVENUE': 60000, 'EXPENSES': 30000, 'PROFIT': 30000}, {'DEPT_NAME': 'Engineering', 'MONTH': 'January', 'REVENUE': 30000, 'EXPENSES': 15000, 'PROFIT': 15000}, {'DEPT_NAME': 'Engineering', 'MONTH': 'February', 'REVENUE': 40000, 'EXPENSES': 20000, 'PROFIT': 20000}, {'DEPT_NAME': 'Engineering', 'MONTH': 'March', 'REVENUE': 50000, 'EXPENSES': 25000, 'PROFIT': 25000}]",
            "summary": "Monthly performance metrics for each department:\n- Sales:\n  - January: 50,000 revenue, 30,000 expenses, 20,000 profit\n  - February: 60,000 revenue, 35,000 expenses, 25,000 profit\n  - March: 70,000 revenue, 40,000 expenses, 30,000 profit\n- Marketing:\n  - January: 40,000 revenue, 20,000 expenses, 20,000 profit\n  - February: 50,000 revenue, 25,000 expenses, 25,000 profit\n  - March: 60,000 revenue, 30,000 expenses, 30,000 profit\n- Engineering:\n  - January: 30,000 revenue, 15,000 expenses, 15,000 profit\n  - February: 40,000 revenue, 20,000 expenses, 20,000 profit\n  - March: 50,000 revenue, 25,000 expenses, 25,000 profit"
        },
        {
            # 600 tokens
            "question": "What are the quarterly sales and number of transactions for each product category?",
            "raw_data": "[{'CATEGORY_NAME': 'Electronics', 'Q1_SALES': 50000, 'Q1_TRANSACTIONS': 300, 'Q2_SALES': 60000, 'Q2_TRANSACTIONS': 350, 'Q3_SALES': 70000, 'Q3_TRANSACTIONS': 400, 'Q4_SALES': 80000, 'Q4_TRANSACTIONS': 450}, {'CATEGORY_NAME': 'Furniture', 'Q1_SALES': 40000, 'Q1_TRANSACTIONS': 250, 'Q2_SALES': 50000, 'Q2_TRANSACTIONS': 300, 'Q3_SALES': 60000, 'Q3_TRANSACTIONS': 350, 'Q4_SALES': 70000, 'Q4_TRANSACTIONS': 400}, {'CATEGORY_NAME': 'Clothing', 'Q1_SALES': 30000, 'Q1_TRANSACTIONS': 200, 'Q2_SALES': 40000, 'Q2_TRANSACTIONS': 250, 'Q3_SALES': 50000, 'Q3_TRANSACTIONS': 300, 'Q4_SALES': 60000, 'Q4_TRANSACTIONS': 350}, {'CATEGORY_NAME': 'Toys', 'Q1_SALES': 20000, 'Q1_TRANSACTIONS': 150, 'Q2_SALES': 30000, 'Q2_TRANSACTIONS': 200, 'Q3_SALES': 40000, 'Q3_TRANSACTIONS': 250, 'Q4_SALES': 50000, 'Q4_TRANSACTIONS': 300}]",
            "summary": "Quarterly sales and number of transactions for each product category:\n- Electronics:\n  - Q1: 50,000 sales, 300 transactions\n  - Q2: 60,000 sales, 350 transactions\n  - Q3: 70,000 sales, 400 transactions\n  - Q4: 80,000 sales, 450 transactions\n- Furniture:\n  - Q1: 40,000 sales, 250 transactions\n  - Q2: 50,000 sales, 300 transactions\n  - Q3: 60,000 sales, 350 transactions\n  - Q4: 70,000 sales, 400 transactions\n- Clothing:\n  - Q1: 30,000 sales, 200 transactions\n  - Q2: 40,000 sales, 250 transactions\n  - Q3: 50,000 sales, 300 transactions\n  - Q4: 60,000 sales, 350 transactions\n- Toys:\n  - Q1: 20,000 sales, 150 transactions\n  - Q2: 30,000 sales, 200 transactions\n  - Q3: 40,000 sales, 250 transactions\n  - Q4: 50,000 sales, 300 transactions"
        },
        {
            # 628 tokens
            "question": "What are the monthly sales figures and transaction counts for each region?",
            "raw_data": "[{'REGION': 'North', 'JAN_SALES': 30000, 'JAN_TRANSACTIONS': 150, 'FEB_SALES': 32000, 'FEB_TRANSACTIONS': 160, 'MAR_SALES': 35000, 'MAR_TRANSACTIONS': 170, 'APR_SALES': 37000, 'APR_TRANSACTIONS': 180, 'MAY_SALES': 40000, 'MAY_TRANSACTIONS': 190, 'JUN_SALES': 42000, 'JUN_TRANSACTIONS': 200}, {'REGION': 'South', 'JAN_SALES': 28000, 'JAN_TRANSACTIONS': 140, 'FEB_SALES': 30000, 'FEB_TRANSACTIONS': 150, 'MAR_SALES': 33000, 'MAR_TRANSACTIONS': 160, 'APR_SALES': 35000, 'APR_TRANSACTIONS': 170, 'MAY_SALES': 38000, 'MAY_TRANSACTIONS': 180, 'JUN_SALES': 40000, 'JUN_TRANSACTIONS': 190}, {'REGION': 'East', 'JAN_SALES': 25000, 'JAN_TRANSACTIONS': 130, 'FEB_SALES': 27000, 'FEB_TRANSACTIONS': 140, 'MAR_SALES': 30000, 'MAR_TRANSACTIONS': 150, 'APR_SALES': 32000, 'APR_TRANSACTIONS': 160, 'MAY_SALES': 35000, 'MAY_TRANSACTIONS': 170, 'JUN_SALES': 37000, 'JUN_TRANSACTIONS': 180}]",
            "summary": "Monthly sales figures and transaction counts for each region:\n- North:\n  - January: 30,000 sales, 150 transactions\n  - February: 32,000 sales, 160 transactions\n  - March: 35,000 sales, 170 transactions\n  - April: 37,000 sales, 180 transactions\n  - May: 40,000 sales, 190 transactions\n  - June: 42,000 sales, 200 transactions\n- South:\n  - January: 28,000 sales, 140 transactions\n  - February: 30,000 sales, 150 transactions\n  - March: 33,000 sales, 160 transactions\n  - April: 35,000 sales, 170 transactions\n  - May: 38,000 sales, 180 transactions\n  - June: 40,000 sales, 190 transactions\n- East:\n  - January: 25,000 sales, 130 transactions\n  - February: 27,000 sales, 140 transactions\n  - March: 30,000 sales, 150 transactions\n  - April: 32,000 sales, 160 transactions\n  - May: 35,000 sales, 170 transactions\n  - June: 37,000 sales, 180 transactions"
        },
        {
            # 524 tokens
            "question": "What are the yearly performance metrics for each sales region, including total sales, total expenses, and net profit?",
            "raw_data": "[{'REGION': 'North America', 'YEAR': 2023, 'TOTAL_SALES': 1500000, 'TOTAL_EXPENSES': 900000, 'NET_PROFIT': 600000}, {'REGION': 'Europe', 'YEAR': 2023, 'TOTAL_SALES': 1300000, 'TOTAL_EXPENSES': 800000, 'NET_PROFIT': 500000}, {'REGION': 'Asia', 'YEAR': 2023, 'TOTAL_SALES': 1200000, 'TOTAL_EXPENSES': 700000, 'NET_PROFIT': 500000}, {'REGION': 'South America', 'YEAR': 2023, 'TOTAL_SALES': 1100000, 'TOTAL_EXPENSES': 650000, 'NET_PROFIT': 450000}, {'REGION': 'Africa', 'YEAR': 2023, 'TOTAL_SALES': 1000000, 'TOTAL_EXPENSES': 600000, 'NET_PROFIT': 400000}, {'REGION': 'Australia', 'YEAR': 2023, 'TOTAL_SALES': 950000, 'TOTAL_EXPENSES': 550000, 'NET_PROFIT': 400000}]",
            "summary": "Yearly performance metrics for each sales region:\n- North America: 1,500,000 total sales, 900,000 total expenses, 600,000 net profit\n- Europe: 1,300,000 total sales, 800,000 total expenses, 500,000 net profit\n- Asia: 1,200,000 total sales, 700,000 total expenses, 500,000 net profit\n- South America: 1,100,000 total sales, 650,000 total expenses, 450,000 net profit\n- Africa: 1,000,000 total sales, 600,000 total expenses, 400,000 net profit\n- Australia: 950,000 total sales, 550,000 total expenses, 400,000 net profit"
        },
        {
            # 611 tokens
            "question": "What are the quarterly performance metrics for each employee in the Engineering department, including total projects handled, hours worked, and client satisfaction ratings?",
            "raw_data": "[{'EMPLOYEE_NAME': 'John Doe', 'Q1_PROJECTS': 5, 'Q1_HOURS': 160, 'Q1_SATISFACTION': 90, 'Q2_PROJECTS': 6, 'Q2_HOURS': 170, 'Q2_SATISFACTION': 92, 'Q3_PROJECTS': 7, 'Q3_HOURS': 180, 'Q3_SATISFACTION': 94, 'Q4_PROJECTS': 8, 'Q4_HOURS': 190, 'Q4_SATISFACTION': 96}, {'EMPLOYEE_NAME': 'Jane Smith', 'Q1_PROJECTS': 4, 'Q1_HOURS': 150, 'Q1_SATISFACTION': 88, 'Q2_PROJECTS': 5, 'Q2_HOURS': 160, 'Q2_SATISFACTION': 89, 'Q3_PROJECTS': 6, 'Q3_HOURS': 170, 'Q3_SATISFACTION': 91, 'Q4_PROJECTS': 7, 'Q4_HOURS': 180, 'Q4_SATISFACTION': 93}, {'EMPLOYEE_NAME': 'Bob Johnson', 'Q1_PROJECTS': 6, 'Q1_HOURS': 165, 'Q1_SATISFACTION': 92, 'Q2_PROJECTS': 7, 'Q2_HOURS': 175, 'Q2_SATISFACTION': 93, 'Q3_PROJECTS': 8, 'Q3_HOURS': 185, 'Q3_SATISFACTION': 95, 'Q4_PROJECTS': 9, 'Q4_HOURS': 195, 'Q4_SATISFACTION': 97}]",
            "summary": "Quarterly performance metrics for each employee in the Engineering department:\n- John Doe:\n  - Q1: 5 projects, 160 hours, 90 satisfaction\n  - Q2: 6 projects, 170 hours, 92 satisfaction\n  - Q3: 7 projects, 180 hours, 94 satisfaction\n  - Q4: 8 projects, 190 hours, 96 satisfaction\n- Jane Smith:\n  - Q1: 4 projects, 150 hours, 88 satisfaction\n  - Q2: 5 projects, 160 hours, 89 satisfaction\n  - Q3: 6 projects, 170 hours, 91 satisfaction\n  - Q4: 7 projects, 180 hours, 93 satisfaction\n- Bob Johnson:\n  - Q1: 6 projects, 165 hours, 92 satisfaction\n  - Q2: 7 projects, 175 hours, 93 satisfaction\n  - Q3: 8 projects, 185 hours, 95 satisfaction\n  - Q4: 9 projects, 195 hours, 97 satisfaction"
        },
        {
            # 555 tokens
            "question": "What are the quarterly revenue and expense figures for each product category?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'Q1_REVENUE': 500000, 'Q1_EXPENSES': 200000, 'Q2_REVENUE': 550000, 'Q2_EXPENSES': 220000, 'Q3_REVENUE': 600000, 'Q3_EXPENSES': 240000, 'Q4_REVENUE': 650000, 'Q4_EXPENSES': 260000}, {'CATEGORY': 'Furniture', 'Q1_REVENUE': 300000, 'Q1_EXPENSES': 120000, 'Q2_REVENUE': 320000, 'Q2_EXPENSES': 130000, 'Q3_REVENUE': 350000, 'Q3_EXPENSES': 140000, 'Q4_REVENUE': 370000, 'Q4_EXPENSES': 150000}, {'CATEGORY': 'Clothing', 'Q1_REVENUE': 400000, 'Q1_EXPENSES': 150000, 'Q2_REVENUE': 420000, 'Q2_EXPENSES': 160000, 'Q3_REVENUE': 450000, 'Q3_EXPENSES': 170000, 'Q4_REVENUE': 480000, 'Q4_EXPENSES': 180000}]",
            "summary": "Quarterly revenue and expense figures for each product category:\n- Electronics:\n  - Q1: 500,000 revenue, 200,000 expenses\n  - Q2: 550,000 revenue, 220,000 expenses\n  - Q3: 600,000 revenue, 240,000 expenses\n  - Q4: 650,000 revenue, 260,000 expenses\n- Furniture:\n  - Q1: 300,000 revenue, 120,000 expenses\n  - Q2: 320,000 revenue, 130,000 expenses\n  - Q3: 350,000 revenue, 140,000 expenses\n  - Q4: 370,000 revenue, 150,000 expenses\n- Clothing:\n  - Q1: 400,000 revenue, 150,000 expenses\n  - Q2: 420,000 revenue, 160,000 expenses\n  - Q3: 450,000 revenue, 170,000 expenses\n  - Q4: 480,000 revenue, 180,000 expenses"
        },
        {
            # 537 tokens
            "question": "What are the annual sales and profit margins for each product in the last two years?",
            "raw_data": "[{'PRODUCT': 'Laptop', 'YEAR_1_SALES': 1000000, 'YEAR_1_PROFIT_MARGIN': 0.25, 'YEAR_2_SALES': 1100000, 'YEAR_2_PROFIT_MARGIN': 0.26}, {'PRODUCT': 'Smartphone', 'YEAR_1_SALES': 1500000, 'YEAR_1_PROFIT_MARGIN': 0.30, 'YEAR_2_SALES': 1600000, 'YEAR_2_PROFIT_MARGIN': 0.31}, {'PRODUCT': 'Tablet', 'YEAR_1_SALES': 800000, 'YEAR_1_PROFIT_MARGIN': 0.20, 'YEAR_2_SALES': 900000, 'YEAR_2_PROFIT_MARGIN': 0.22}, {'PRODUCT': 'Headphones', 'YEAR_1_SALES': 500000, 'YEAR_1_PROFIT_MARGIN': 0.15, 'YEAR_2_SALES': 600000, 'YEAR_2_PROFIT_MARGIN': 0.16}, {'PRODUCT': 'Smartwatch', 'YEAR_1_SALES': 700000, 'YEAR_1_PROFIT_MARGIN': 0.18, 'YEAR_2_SALES': 750000, 'YEAR_2_PROFIT_MARGIN': 0.19}]",
            "summary": "Annual sales and profit margins for each product in the last two years:\n- Laptop:\n  - Year 1: 1,000,000 sales, 25% profit margin\n  - Year 2: 1,100,000 sales, 26% profit margin\n- Smartphone:\n  - Year 1: 1,500,000 sales, 30% profit margin\n  - Year 2: 1,600,000 sales, 31% profit margin\n- Tablet:\n  - Year 1: 800,000 sales, 20% profit margin\n  - Year 2: 900,000 sales, 22% profit margin\n- Headphones:\n  - Year 1: 500,000 sales, 15% profit margin\n  - Year 2: 600,000 sales, 16% profit margin\n- Smartwatch:\n  - Year 1: 700,000 sales, 18% profit margin\n  - Year 2: 750,000 sales, 19% profit margin"
        },
        {
            # 707 tokens
            "question": "What are the monthly revenue figures, profit margins, and customer feedback scores for each service category over the past six months?",
            "raw_data": "[{'CATEGORY': 'Consulting', 'JAN_REVENUE': 60000, 'JAN_PROFIT_MARGIN': 0.3, 'JAN_FEEDBACK': 88, 'FEB_REVENUE': 62000, 'FEB_PROFIT_MARGIN': 0.32, 'FEB_FEEDBACK': 89, 'MAR_REVENUE': 63000, 'MAR_PROFIT_MARGIN': 0.31, 'MAR_FEEDBACK': 90, 'APR_REVENUE': 65000, 'APR_PROFIT_MARGIN': 0.33, 'APR_FEEDBACK': 91, 'MAY_REVENUE': 66000, 'MAY_PROFIT_MARGIN': 0.34, 'MAY_FEEDBACK': 92, 'JUN_REVENUE': 67000, 'JUN_PROFIT_MARGIN': 0.35, 'JUN_FEEDBACK': 93}, {'CATEGORY': 'Support', 'JAN_REVENUE': 40000, 'JAN_PROFIT_MARGIN': 0.2, 'JAN_FEEDBACK': 85, 'FEB_REVENUE': 42000, 'FEB_PROFIT_MARGIN': 0.22, 'FEB_FEEDBACK': 86, 'MAR_REVENUE': 43000, 'MAR_PROFIT_MARGIN': 0.21, 'MAR_FEEDBACK': 87, 'APR_REVENUE': 45000, 'APR_PROFIT_MARGIN': 0.23, 'APR_FEEDBACK': 88, 'MAY_REVENUE': 46000, 'MAY_PROFIT_MARGIN': 0.24, 'MAY_FEEDBACK': 89, 'JUN_REVENUE': 47000, 'JUN_PROFIT_MARGIN': 0.25, 'JUN_FEEDBACK': 90}]",
            "summary": "Monthly revenue figures, profit margins, and customer feedback scores for each service category over the past six months:\n- Consulting:\n  - January: 60,000 revenue, 30% profit margin, 88 feedback\n  - February: 62,000 revenue, 32% profit margin, 89 feedback\n  - March: 63,000 revenue, 31% profit margin, 90 feedback\n  - April: 65,000 revenue, 33% profit margin, 91 feedback\n  - May: 66,000 revenue, 34% profit margin, 92 feedback\n  - June: 67,000 revenue, 35% profit margin, 93 feedback\n- Support:\n  - January: 40,000 revenue, 20% profit margin, 85 feedback\n  - February: 42,000 revenue, 22% profit margin, 86 feedback\n  - March: 43,000 revenue, 21% profit margin, 87 feedback\n  - April: 45,000 revenue, 23% profit margin, 88 feedback\n  - May: 46,000 revenue, 24% profit margin, 89 feedback\n  - June: 47,000 revenue, 25% profit margin, 90 feedback"
        },
        {
            # 694 tokens
            "question": "What are the quarterly sales figures, profit margins, and employee performance ratings for each sales team in the company over the past six months?",
            "raw_data": "[{'TEAM': 'Alpha', 'Q1_SALES': 80000, 'Q1_PROFIT_MARGIN': 0.25, 'Q1_PERFORMANCE': 90, 'Q2_SALES': 85000, 'Q2_PROFIT_MARGIN': 0.27, 'Q2_PERFORMANCE': 91, 'Q3_SALES': 90000, 'Q3_PROFIT_MARGIN': 0.26, 'Q3_PERFORMANCE': 92}, {'TEAM': 'Beta', 'Q1_SALES': 75000, 'Q1_PROFIT_MARGIN': 0.22, 'Q1_PERFORMANCE': 88, 'Q2_SALES': 78000, 'Q2_PROFIT_MARGIN': 0.24, 'Q2_PERFORMANCE': 89, 'Q3_SALES': 81000, 'Q3_PROFIT_MARGIN': 0.23, 'Q3_PERFORMANCE': 90}, {'TEAM': 'Gamma', 'Q1_SALES': 70000, 'Q1_PROFIT_MARGIN': 0.2, 'Q1_PERFORMANCE': 85, 'Q2_SALES': 73000, 'Q2_PROFIT_MARGIN': 0.22, 'Q2_PERFORMANCE': 86, 'Q3_SALES': 76000, 'Q3_PROFIT_MARGIN': 0.21, 'Q3_PERFORMANCE': 87}, {'TEAM': 'Delta', 'Q1_SALES': 65000, 'Q1_PROFIT_MARGIN': 0.18, 'Q1_PERFORMANCE': 83, 'Q2_SALES': 68000, 'Q2_PROFIT_MARGIN': 0.2, 'Q2_PERFORMANCE': 84, 'Q3_SALES': 71000, 'Q3_PROFIT_MARGIN': 0.19, 'Q3_PERFORMANCE': 85}]",
            "summary": "Quarterly sales figures, profit margins, and employee performance ratings for each sales team over the past six months:\n- Alpha:\n  - Q1: 80,000 sales, 25% profit margin, 90 performance\n  - Q2: 85,000 sales, 27% profit margin, 91 performance\n  - Q3: 90,000 sales, 26% profit margin, 92 performance\n- Beta:\n  - Q1: 75,000 sales, 22% profit margin, 88 performance\n  - Q2: 78,000 sales, 24% profit margin, 89 performance\n  - Q3: 81,000 sales, 23% profit margin, 90 performance\n- Gamma:\n  - Q1: 70,000 sales, 20% profit margin, 85 performance\n  - Q2: 73,000 sales, 22% profit margin, 86 performance\n  - Q3: 76,000 sales, 21% profit margin, 87 performance\n- Delta:\n  - Q1: 65,000 sales, 18% profit margin, 83 performance\n  - Q2: 68,000 sales, 20% profit margin, 84 performance\n  - Q3: 71,000 sales, 19% profit margin, 85 performance"
        },
        {
            # 503 tokens
            "question": "What are the bi-weekly production metrics for each manufacturing unit, including units produced, units rejected, and average production time?",
            "raw_data": "[{'UNIT': 'A', 'WEEK1_PROD': 5000, 'WEEK1_REJECT': 50, 'WEEK1_TIME': 8.5, 'WEEK2_PROD': 5200, 'WEEK2_REJECT': 45, 'WEEK2_TIME': 8.3, 'WEEK3_PROD': 5400, 'WEEK3_REJECT': 40, 'WEEK3_TIME': 8.1}, {'UNIT': 'B', 'WEEK1_PROD': 4800, 'WEEK1_REJECT': 60, 'WEEK1_TIME': 8.7, 'WEEK2_PROD': 5000, 'WEEK2_REJECT': 55, 'WEEK2_TIME': 8.6, 'WEEK3_PROD': 5100, 'WEEK3_REJECT': 50, 'WEEK3_TIME': 8.4}, {'UNIT': 'C', 'WEEK1_PROD': 4600, 'WEEK1_REJECT': 70, 'WEEK1_TIME': 8.9, 'WEEK2_PROD': 4800, 'WEEK2_REJECT': 65, 'WEEK2_TIME': 8.8, 'WEEK3_PROD': 4900, 'WEEK3_REJECT': 60, 'WEEK3_TIME': 8.5}]",
            "summary": "Bi-weekly production metrics for each manufacturing unit:\n- Unit A:\n  - Week 1: 5,000 units produced, 50 units rejected, 8.5 hours average production time\n  - Week 2: 5,200 units produced, 45 units rejected, 8.3 hours average production time\n  - Week 3: 5,400 units produced, 40 units rejected, 8.1 hours average production time\n- Unit B:\n  - Week 1: 4,800 units produced, 60 units rejected, 8.7 hours average production time\n  - Week 2: 5,000 units produced, 55 units rejected, 8.6 hours average production time\n  - Week 3: 5,100 units produced, 50 units rejected, 8.4 hours average production time\n- Unit C:\n  - Week 1: 4,600 units produced, 70 units rejected, 8.9 hours average production time\n  - Week 2: 4,800 units produced, 65 units rejected, 8.8 hours average production time\n  - Week 3: 4,900 units produced, 60 units rejected, 8.5 hours average production time"
        },
        {
            # 572 tokens
            "question": "What are the monthly financial performance metrics for each department, including total revenue, total expenses, and net profit?",
            "raw_data": "[{'DEPARTMENT': 'Finance', 'JAN_REVENUE': 90000, 'JAN_EXPENSES': 50000, 'JAN_PROFIT': 40000, 'FEB_REVENUE': 95000, 'FEB_EXPENSES': 52000, 'FEB_PROFIT': 43000, 'MAR_REVENUE': 100000, 'MAR_EXPENSES': 54000, 'MAR_PROFIT': 46000}, {'DEPARTMENT': 'HR', 'JAN_REVENUE': 80000, 'JAN_EXPENSES': 48000, 'JAN_PROFIT': 32000, 'FEB_REVENUE': 85000, 'FEB_EXPENSES': 49000, 'FEB_PROFIT': 36000, 'MAR_REVENUE': 90000, 'MAR_EXPENSES': 50000, 'MAR_PROFIT': 40000}, {'DEPARTMENT': 'IT', 'JAN_REVENUE': 100000, 'JAN_EXPENSES': 60000, 'JAN_PROFIT': 40000, 'FEB_REVENUE': 105000, 'FEB_EXPENSES': 62000, 'FEB_PROFIT': 43000, 'MAR_REVENUE': 110000, 'MAR_EXPENSES': 64000, 'MAR_PROFIT': 46000}]",
            "summary": "Monthly financial performance metrics for each department:\n- Finance:\n  - January: 90,000 revenue, 50,000 expenses, 40,000 net profit\n  - February: 95,000 revenue, 52,000 expenses, 43,000 net profit\n  - March: 100,000 revenue, 54,000 expenses, 46,000 net profit\n- HR:\n  - January: 80,000 revenue, 48,000 expenses, 32,000 net profit\n  - February: 85,000 revenue, 49,000 expenses, 36,000 net profit\n  - March: 90,000 revenue, 50,000 expenses, 40,000 net profit\n- IT:\n  - January: 100,000 revenue, 60,000 expenses, 40,000 net profit\n  - February: 105,000 revenue, 62,000 expenses, 43,000 net profit\n  - March: 110,000 revenue, 64,000 expenses, 46,000 net profit"
        },
        {
            # 582 tokens
            "question": "What are the monthly revenue and expense breakdowns for each branch, including net profit and percentage growth from the previous month?",
            "raw_data": "[{'BRANCH': 'New York', 'JAN_REVENUE': 150000, 'JAN_EXPENSES': 80000, 'JAN_PROFIT': 70000, 'FEB_REVENUE': 160000, 'FEB_EXPENSES': 85000, 'FEB_PROFIT': 75000, 'MAR_REVENUE': 170000, 'MAR_EXPENSES': 90000, 'MAR_PROFIT': 80000}, {'BRANCH': 'Los Angeles', 'JAN_REVENUE': 140000, 'JAN_EXPENSES': 75000, 'JAN_PROFIT': 65000, 'FEB_REVENUE': 150000, 'FEB_EXPENSES': 80000, 'FEB_PROFIT': 70000, 'MAR_REVENUE': 160000, 'MAR_EXPENSES': 85000, 'MAR_PROFIT': 75000}, {'BRANCH': 'Chicago', 'JAN_REVENUE': 130000, 'JAN_EXPENSES': 70000, 'JAN_PROFIT': 60000, 'FEB_REVENUE': 140000, 'FEB_EXPENSES': 75000, 'FEB_PROFIT': 65000, 'MAR_REVENUE': 150000, 'MAR_EXPENSES': 80000, 'MAR_PROFIT': 70000}]",
            "summary": "Monthly revenue and expense breakdowns for each branch:\n- New York:\n  - January: 150,000 revenue, 80,000 expenses, 70,000 net profit\n  - February: 160,000 revenue, 85,000 expenses, 75,000 net profit\n  - March: 170,000 revenue, 90,000 expenses, 80,000 net profit\n- Los Angeles:\n  - January: 140,000 revenue, 75,000 expenses, 65,000 net profit\n  - February: 150,000 revenue, 80,000 expenses, 70,000 net profit\n  - March: 160,000 revenue, 85,000 expenses, 75,000 net profit\n- Chicago:\n  - January: 130,000 revenue, 70,000 expenses, 60,000 net profit\n  - February: 140,000 revenue, 75,000 expenses, 65,000 net profit\n  - March: 150,000 revenue, 80,000 expenses, 70,000 net profit"
        },
        {
            # 516 tokens
            "question": "What are the monthly sales figures, returns, and customer satisfaction ratings for each product category?",
            "raw_data": "[{'CATEGORY': 'Electronics', 'JAN_SALES': 50000, 'JAN_RETURNS': 200, 'JAN_SATISFACTION': 90, 'FEB_SALES': 52000, 'FEB_RETURNS': 220, 'FEB_SATISFACTION': 88, 'MAR_SALES': 54000, 'MAR_RETURNS': 240, 'MAR_SATISFACTION': 87}, {'CATEGORY': 'Clothing', 'JAN_SALES': 40000, 'JAN_RETURNS': 300, 'JAN_SATISFACTION': 85, 'FEB_SALES': 42000, 'FEB_RETURNS': 320, 'FEB_SATISFACTION': 84, 'MAR_SALES': 44000, 'MAR_RETURNS': 340, 'MAR_SATISFACTION': 83}, {'CATEGORY': 'Furniture', 'JAN_SALES': 30000, 'JAN_RETURNS': 100, 'JAN_SATISFACTION': 92, 'FEB_SALES': 32000, 'FEB_RETURNS': 120, 'FEB_SATISFACTION': 91, 'MAR_SALES': 34000, 'MAR_RETURNS': 140, 'MAR_SATISFACTION': 90}]",
            "summary": "Monthly sales figures, returns, and customer satisfaction ratings for each product category:\n- Electronics:\n  - January: 50,000 sales, 200 returns, 90% satisfaction\n  - February: 52,000 sales, 220 returns, 88% satisfaction\n  - March: 54,000 sales, 240 returns, 87% satisfaction\n- Clothing:\n  - January: 40,000 sales, 300 returns, 85% satisfaction\n  - February: 42,000 sales, 320 returns, 84% satisfaction\n  - March: 44,000 sales, 340 returns, 83% satisfaction\n- Furniture:\n  - January: 30,000 sales, 100 returns, 92% satisfaction\n  - February: 32,000 sales, 120 returns, 91% satisfaction\n  - March: 34,000 sales, 140 returns, 90% satisfaction"
        }
    ]
    return benchmark_data

def evaluate_text_to_chat(llm, benchmark_data, difficulty):
    """Evaluate the Text-to-Chat LLM using benchmark data."""
    total_queries = len(benchmark_data)
    rouge_l_scores = []
    bert_scores = []
    query_times = []
    rouge = Rouge()

    # Difficulty and Query Clarification
    print(f"\n#################\nBenchmarking {total_queries} number of queries for {difficulty} difficulty\n#################")

    for data in benchmark_data:
        question = data['question']
        raw_data = data['raw_data']
        expected_summary = data['summary']

        # Default variable reset
        temp_rouge_l_f1 = 0
        temp_bert_score = 0
        start_time = time.time()
        print(f"\n\nQuestion: {question}")

        try:
            # Generate textual summary using the QLLM
            generated_summary = LLMConfiguration.generate_textual_insights(llm, question, raw_data)
            generated_summary = str(generated_summary).strip()

            # Time logging
            end_time = time.time()
            query_time = end_time - start_time
            query_times.append(query_time)

            # Evaluate using BERT
            P, R, F1 = score([generated_summary], [expected_summary], lang="en", verbose=True)
            temp_bert_score = np.mean(F1.cpu().numpy())
            bert_scores.append(temp_bert_score)
            # Evaluation using Rouge
            temp_rouge_score = rouge.get_scores(generated_summary, expected_summary, avg=True)
            temp_rouge_l_f1 = temp_rouge_score['rouge-l']['f']
            rouge_l_scores.append(temp_rouge_l_f1)

            print(f"Generated Summary: {generated_summary}")
            print(f"Expected Summary: {expected_summary}")
            print(f"BERT F1 Scoring: {temp_bert_score}")
            print(f"Rouge F1 Scoring: {temp_rouge_l_f1}")
            # print(f"Query Time: {query_time} seconds\n")

        except Exception as e:
            print(f"Error with summary generation: {e}")

        # Store CHAT stat for test case to "chat_benchmark_stats"
        DatabaseConfiguration.store_chat_stat(question, temp_bert_score, temp_rouge_l_f1, difficulty)
        print("Added CHAT stat to chat_benchmark_stats")

    # Calculate a summary of metrics for CHAT difficulty
    average_rouge_score = np.mean(rouge_l_scores)
    average_bert_score = np.mean(bert_scores)
    average_latency = np.mean(query_times)

    print(f"Average Rouge F1: {average_rouge_score:.2f}")
    print(f"Average BERTScore F1: {average_bert_score:.2f}")
    print(f"Average Query Time: {average_latency:.2f} seconds")

    # Store CHAT summary to "chat_summary_stats"
    DatabaseConfiguration.store_chat_summary(difficulty, average_bert_score, average_rouge_score, average_latency)
    print("Added CHAT summary stat to chat_summary_stats")

def run_chat_benchmark():
    # Deploy the Text-to-Chat LLM
    chat_llm = LLMConfiguration.deploy_chat_llama()

    # Load benchmark dataset
    easy_benchmark_data = load_easy_benchmark()
    medium_benchmark_data = load_medium_benchmark()
    hard_benchmark_data = load_hard_benchmark()

    # Evaluate the Text-to-Chat LLM
    evaluate_text_to_chat(chat_llm, easy_benchmark_data, "easy")
    evaluate_text_to_chat(chat_llm, medium_benchmark_data, "medium")
    evaluate_text_to_chat(chat_llm, hard_benchmark_data ,"hard")
