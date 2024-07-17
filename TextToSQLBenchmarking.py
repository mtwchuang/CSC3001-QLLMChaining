# Local libraries
import LLMConfiguration
import DatabaseConfiguration  # Assuming your database functions are in db_config.py
# Others
import time
import sqlite3
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure nltk resources are downloaded
nltk.download('punkt')

# 15 easy queries
def load_easy_benchmark_dataset():
    """Load benchmark dataset of 25 easy natural language questions and their corresponding SQL queries."""
    benchmark_data = [
        # Easy Queries
        {"question": "Show all employee names.", "sql": "SELECT EMPLOYEE_NAME FROM LU_EMPLOYEE"},
        {"question": "List all department names.", "sql": "SELECT DEPT_NAME FROM LU_DEPARTMENT"},
        {"question": "Show all vendor full names.", "sql": "SELECT VENDOR_FULLNAME FROM LU_VENDOR"},
        {"question": "Show all product names.", "sql": "SELECT PRODUCT_NAME FROM LU_PRODUCT"},
        {"question": "List all employee IDs.", "sql": "SELECT EMPLOYEE_ID FROM LU_EMPLOYEE"},
        {"question": "List all vendor IDs.", "sql": "SELECT VENDOR_ID FROM LU_VENDOR"},
        {"question": "Show all department IDs.", "sql": "SELECT DEPT_ID FROM LU_DEPARTMENT"},
        {"question": "Show all product IDs.", "sql": "SELECT PRODUCT_ID FROM LU_PRODUCT"},
        {"question": "List all invoice IDs.", "sql": "SELECT INVOICE_ID FROM FT_INVOICE"},
        {"question": "Show all payment amounts in invoices.", "sql": "SELECT PAYMENT_AMOUNT FROM FT_INVOICE"},
        {"question": "Show all employee names in alphabetical order.", "sql": "SELECT EMPLOYEE_NAME FROM LU_EMPLOYEE ORDER BY EMPLOYEE_NAME"},
        {"question": "How many departments are there?", "sql": "SELECT COUNT(*) FROM LU_DEPARTMENT"},
        {"question": "Show all product names in descending order.", "sql": "SELECT PRODUCT_NAME FROM LU_PRODUCT ORDER BY PRODUCT_NAME DESC"},
        # No results
        # {"question": "Show all vendor full names from the USA.", "sql": "SELECT VENDOR_FULLNAME FROM LU_VENDOR WHERE COUNTRY_CODE = 'USA'"},
        {"question": "List the distinct countries of vendors.", "sql": "SELECT DISTINCT COUNTRY_CODE FROM LU_VENDOR"},
        # No results
        # {"question": "Show all department names containing 'Sales'.", "sql": "SELECT DEPT_NAME FROM LU_DEPARTMENT WHERE DEPT_NAME LIKE '%Sales%'"},         
        {"question": "Show all employee names starting with 'A'.", "sql": "SELECT EMPLOYEE_NAME FROM LU_EMPLOYEE WHERE EMPLOYEE_NAME LIKE 'A%'"},
        # {"question": "How many products are there?", "sql": "SELECT COUNT(*) FROM LU_PRODUCT"},
        # {"question": "List all vendor IDs in ascending order.", "sql": "SELECT VENDOR_ID FROM LU_VENDOR ORDER BY VENDOR_ID"},
        # {"question": "Show all distinct payment amounts in invoices.", "sql": "SELECT DISTINCT PAYMENT_AMOUNT FROM FT_INVOICE"},
        # {"question": "Show all employee names and their IDs.", "sql": "SELECT EMPLOYEE_NAME, EMPLOYEE_ID FROM LU_EMPLOYEE"},
        # {"question": "Show names of departments with IDs less than 500.", "sql": "SELECT DEPT_NAME FROM LU_DEPARTMENT WHERE DEPT_ID < 500"},
        # No results
        # {"question": "List all product names that start with 'Pro'.", "sql": "SELECT PRODUCT_NAME FROM LU_PRODUCT WHERE PRODUCT_NAME LIKE 'Pro%'"},
        # {"question": "Show all vendors sorted by their full names.", "sql": "SELECT VENDOR_FULLNAME FROM LU_VENDOR ORDER BY VENDOR_FULLNAME"},
        # {"question": "Show the names and IDs of all departments.", "sql": "SELECT DEPT_NAME, DEPT_ID FROM LU_DEPARTMENT"}
    ]
    return benchmark_data

# 15 medium queries
def load_medium_benchmark_dataset():
    """Load benchmark dataset of 25 medium natural language questions and their corresponding SQL queries."""
    benchmark_data = [
        {"question": "List all invoices with payment amounts greater than 50000 and department ID lesser than 500.", "sql": "SELECT * FROM FT_INVOICE WHERE PAYMENT_AMOUNT > 50000 AND DEPT_ID < 500"},
        {"question": "Show employee names along with their department names.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME, LU_DEPARTMENT.DEPT_NAME FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID JOIN LU_DEPARTMENT ON FT_INVOICE.DEPT_ID = LU_DEPARTMENT.DEPT_ID"},
        {"question": "Show product names and their total sales amount.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME"},
        {"question": "Show vendor names and their total payment amounts.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME"},
        {"question": "Show employee names who have handled more than 5 invoices.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME HAVING COUNT(FT_INVOICE.INVOICE_ID) > 5"},
        {"question": "Show department names along with the number of employees in each.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME, COUNT(DISTINCT LU_EMPLOYEE.EMPLOYEE_ID) FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID JOIN LU_EMPLOYEE ON FT_INVOICE.EMPLOYEE_ID = LU_EMPLOYEE.EMPLOYEE_ID GROUP BY LU_DEPARTMENT.DEPT_NAME"},
        {"question": "Show the average payment amount for each department.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME, AVG(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME"},
        {"question": "Show vendor names with more than 10 invoices.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME HAVING COUNT(FT_INVOICE.INVOICE_ID) > 10"},
        {"question": "Show product names along with the number of invoices for each product.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, COUNT(FT_INVOICE.INVOICE_ID) FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME"},
        {"question": "Show the total payment amount for each vendor.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME"},
        {"question": "Show the total payment amount for each product.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME"},
        {"question": "Show the number of invoices for each employee.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME, COUNT(FT_INVOICE.INVOICE_ID) FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME"},
        {"question": "Show the highest payment amount for each department.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME, MAX(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME"},
        {"question": "Show the lowest payment amount for each vendor.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, MIN(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME"},
        {"question": "Show the average payment amount for each product.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, AVG(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME"},
        # {"question": "Show department names with more than 5 employees.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID JOIN LU_EMPLOYEE ON FT_INVOICE.EMPLOYEE_ID = LU_EMPLOYEE.EMPLOYEE_ID GROUP BY LU_DEPARTMENT.DEPT_NAME HAVING COUNT(DISTINCT LU_EMPLOYEE.EMPLOYEE_ID) > 5"},
        # {"question": "Show the number of products sold by each department.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME, COUNT(FT_INVOICE.PRODUCT_ID) FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME"},
        # {"question": "Show the names and total payment amounts of vendors with payment amounts greater than the average.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME HAVING SUM(FT_INVOICE.PAYMENT_AMOUNT) > (SELECT AVG(PAYMENT_AMOUNT) FROM FT_INVOICE)"},
        # {"question": "Show the total number of invoices for each product.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, COUNT(FT_INVOICE.INVOICE_ID) FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME"},
        # {"question": "Show the total payment amount for each employee.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME"},
        # {"question": "Show the names and payment amounts of employees who have handled invoices with payment amounts greater than the average payment amount.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME, FT_INVOICE.PAYMENT_AMOUNT FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID WHERE FT_INVOICE.PAYMENT_AMOUNT > (SELECT AVG(PAYMENT_AMOUNT) FROM FT_INVOICE)"},
        # {"question": "Show the names of products and the total payment amounts for those products where the total is greater than 10000.", "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_PAYMENT FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME HAVING SUM(FT_INVOICE.PAYMENT_AMOUNT) > 10000"},
        # {"question": "Show vendor names and the number of invoices from each vendor.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, COUNT(FT_INVOICE.INVOICE_ID) FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME"},
        # {"question": "Show the department names with the highest total payment amount.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME ORDER BY SUM(FT_INVOICE.PAYMENT_AMOUNT) DESC LIMIT 1"},
        # {"question": "Show the employee name who has handled the highest number of invoices.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME ORDER BY COUNT(FT_INVOICE.INVOICE_ID) DESC LIMIT 1"},
        # {"question": "Show the department names that have generated more than $10,000 in total sales.", "sql": "SELECT LU_DEPARTMENT.DEPT_NAME FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME HAVING SUM(FT_INVOICE.PAYMENT_AMOUNT) > 10000"},
        # {"question": "Show the names of employees who have handled invoices with an average payment amount greater than $5000.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME HAVING AVG(FT_INVOICE.PAYMENT_AMOUNT) > 5000"},
        # {"question": "Retrieve the names of vendors that have invoices with payment amounts less than 1000 and more than 20000.", "sql": "SELECT DISTINCT LU_VENDOR.VENDOR_FULLNAME FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID WHERE FT_INVOICE.PAYMENT_AMOUNT < 1000 OR FT_INVOICE.PAYMENT_AMOUNT > 20000"},
        # {"question": "Show the names of employees and the total payment amounts of invoices they have handled, excluding those employees who have handled less than 3 invoices.", "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_PAYMENT FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME HAVING COUNT(FT_INVOICE.INVOICE_ID) >= 3"},
        # {"question": "Show the vendor names who have supplied the most expensive product.", "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID JOIN LU_PRODUCT ON FT_INVOICE.PRODUCT_ID = LU_PRODUCT.PRODUCT_ID WHERE FT_INVOICE.PAYMENT_AMOUNT = (SELECT MAX(PAYMENT_AMOUNT) FROM FT_INVOICE)"},
    ]
    return benchmark_data

# 15 hard queries
def load_hard_benchmark_dataset():
    """Load benchmark dataset of 15 hard natural language questions and their corresponding SQL queries."""
    benchmark_data = [
        {
            "question": "Show the products and their respective departments that have the highest sales in each department.", 
            "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, LU_DEPARTMENT.DEPT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_SALES FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID JOIN LU_DEPARTMENT ON FT_INVOICE.DEPT_ID = LU_DEPARTMENT.DEPT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME, LU_DEPARTMENT.DEPT_NAME ORDER BY LU_DEPARTMENT.DEPT_NAME, TOTAL_SALES DESC"
        },
        {
            "question": "Show the names of vendors and the total payment amounts for those vendors where the total is greater than the average of all vendors.", 
            "sql": "WITH VendorPayments AS (SELECT LU_VENDOR.VENDOR_FULLNAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_PAYMENT FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME), AveragePayment AS (SELECT AVG(TOTAL_PAYMENT) AS AVG_TOTAL_PAYMENT FROM VendorPayments) SELECT VP.VENDOR_FULLNAME, VP.TOTAL_PAYMENT FROM VendorPayments VP JOIN AveragePayment AP ON VP.TOTAL_PAYMENT > AP.AVG_TOTAL_PAYMENT"
        },
        # no result
        {
            "question": "Show the names and total sales amounts of products that contribute to more than 10% of the total sales in their respective departments.", 
            "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, LU_DEPARTMENT.DEPT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_SALES FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID JOIN LU_DEPARTMENT ON FT_INVOICE.DEPT_ID = LU_DEPARTMENT.DEPT_ID GROUP BY LU_PRODUCT.PRODUCT_NAME, LU_DEPARTMENT.DEPT_NAME HAVING TOTAL_SALES > 0.1 * (SELECT SUM(PAYMENT_AMOUNT) FROM FT_INVOICE WHERE DEPT_ID = LU_DEPARTMENT.DEPT_ID)"
        },
        # no result
        {
            "question": "Show the department names and their total sales amounts where the department's total sales are more than the average department sales.", 
            "sql": "WITH DepartmentSales AS (SELECT LU_DEPARTMENT.DEPT_NAME, SUM(FT_INVOICE.PAYMENT_AMOUNT) AS TOTAL_SALES FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME) SELECT DS.DEPT_NAME, DS.TOTAL_SALES FROM DepartmentSales DS WHERE DS.TOTAL_SALES > (SELECT AVG(TOTAL_SALES) FROM DepartmentSales)"
        },
        {
            "question": "List the vendor names and the average payment amounts for invoices that are higher than the average payment amount of all invoices.", 
            "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, AVG(FT_INVOICE.PAYMENT_AMOUNT) AS AVG_PAYMENT FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME HAVING AVG_PAYMENT > (SELECT AVG(PAYMENT_AMOUNT) FROM FT_INVOICE)"
        },
        {
            "question": "Show the vendor names who have supplied products to more than three different departments.", 
            "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME HAVING COUNT(DISTINCT FT_INVOICE.DEPT_ID) > 3"
        },
        # No result
        {
            "question": "Show the employee names who have managed invoices for both 'Sales' and 'Marketing' departments.", 
            "sql": "SELECT EMPLOYEE_NAME FROM LU_EMPLOYEE WHERE EMPLOYEE_ID IN (SELECT EMPLOYEE_ID FROM FT_INVOICE WHERE DEPT_ID = (SELECT DEPT_ID FROM LU_DEPARTMENT WHERE DEPT_NAME = 'Sales')) AND EMPLOYEE_ID IN (SELECT EMPLOYEE_ID FROM FT_INVOICE WHERE DEPT_ID = (SELECT DEPT_ID FROM LU_DEPARTMENT WHERE DEPT_NAME = 'Marketing'))"
        },
        {
            "question": "Show the product names sold by the highest number of vendors.", 
            "sql": "SELECT PRODUCT_NAME FROM LU_PRODUCT WHERE PRODUCT_ID = (SELECT PRODUCT_ID FROM FT_INVOICE GROUP BY PRODUCT_ID ORDER BY COUNT(DISTINCT VENDOR_ID) DESC LIMIT 1)"
        },
        {
            "question": "Show the vendor names with the highest average invoice payment amount.", 
            "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID GROUP BY LU_VENDOR.VENDOR_FULLNAME ORDER BY AVG(FT_INVOICE.PAYMENT_AMOUNT) DESC LIMIT 1"
        },
        # No result
        {
            "question": "Show the employee names who have managed invoices worth more than the total sales of the 'Sales' department.", 
            "sql": "SELECT LU_EMPLOYEE.EMPLOYEE_NAME FROM LU_EMPLOYEE JOIN FT_INVOICE ON LU_EMPLOYEE.EMPLOYEE_ID = FT_INVOICE.EMPLOYEE_ID GROUP BY LU_EMPLOYEE.EMPLOYEE_NAME HAVING SUM(FT_INVOICE.PAYMENT_AMOUNT) > (SELECT SUM(PAYMENT_AMOUNT) FROM FT_INVOICE WHERE DEPT_ID = (SELECT DEPT_ID FROM LU_DEPARTMENT WHERE DEPT_NAME = 'Sales'))"
        },
        {
            "question": "Show the department names with sales more than 50% of the maximum department sales.", 
            "sql": "SELECT LU_DEPARTMENT.DEPT_NAME FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME HAVING SUM(FT_INVOICE.PAYMENT_AMOUNT) > 0.5 * (SELECT MAX(TOTAL_SALES) FROM (SELECT SUM(PAYMENT_AMOUNT) AS TOTAL_SALES FROM FT_INVOICE GROUP BY DEPT_ID))"
        },
        {
            "question": "Show the product names, department names, and vendor names for products that have been sold by vendors in departments with sales greater than 10000.",
            "sql": "SELECT LU_PRODUCT.PRODUCT_NAME, LU_DEPARTMENT.DEPT_NAME, LU_VENDOR.VENDOR_FULLNAME FROM LU_PRODUCT JOIN FT_INVOICE ON LU_PRODUCT.PRODUCT_ID = FT_INVOICE.PRODUCT_ID JOIN LU_DEPARTMENT ON FT_INVOICE.DEPT_ID = LU_DEPARTMENT.DEPT_ID JOIN LU_VENDOR ON FT_INVOICE.VENDOR_ID = LU_VENDOR.VENDOR_ID WHERE LU_DEPARTMENT.DEPT_ID IN (SELECT DEPT_ID FROM FT_INVOICE GROUP BY DEPT_ID HAVING SUM(PAYMENT_AMOUNT) > 10000)"
        },
        {
            "question": "Show the vendor names and their average payment amounts for vendors who have at least 5 invoices with payments greater than 1000.",
            "sql": "SELECT LU_VENDOR.VENDOR_FULLNAME, AVG(FT_INVOICE.PAYMENT_AMOUNT) AS AVG_PAYMENT FROM LU_VENDOR JOIN FT_INVOICE ON LU_VENDOR.VENDOR_ID = FT_INVOICE.VENDOR_ID WHERE FT_INVOICE.PAYMENT_AMOUNT > 1000 GROUP BY LU_VENDOR.VENDOR_FULLNAME HAVING COUNT(FT_INVOICE.INVOICE_ID) >= 5"
        },
        {
            "question": "Show the employee names and their total invoice amounts, but only for employees whose total invoice amount is greater than the average total invoice amount of all employees.",
            "sql": "WITH EmployeeTotals AS (SELECT EMPLOYEE_ID, SUM(PAYMENT_AMOUNT) AS TOTAL_AMOUNT FROM FT_INVOICE GROUP BY EMPLOYEE_ID), AverageTotal AS (SELECT AVG(TOTAL_AMOUNT) AS AVG_TOTAL FROM EmployeeTotals) SELECT LU_EMPLOYEE.EMPLOYEE_NAME, EmployeeTotals.TOTAL_AMOUNT FROM LU_EMPLOYEE JOIN EmployeeTotals ON LU_EMPLOYEE.EMPLOYEE_ID = EmployeeTotals.EMPLOYEE_ID WHERE EmployeeTotals.TOTAL_AMOUNT > (SELECT AVG_TOTAL FROM AverageTotal)"
        },
        # no result
        {
            "question": "Show the department names that have more than twice the average number of invoices across all departments.",
            "sql": "SELECT LU_DEPARTMENT.DEPT_NAME FROM LU_DEPARTMENT JOIN FT_INVOICE ON LU_DEPARTMENT.DEPT_ID = FT_INVOICE.DEPT_ID GROUP BY LU_DEPARTMENT.DEPT_NAME HAVING COUNT(FT_INVOICE.INVOICE_ID) > 2 * (SELECT AVG(DeptInvoiceCount.TOTAL_COUNT) FROM (SELECT DEPT_ID, COUNT(INVOICE_ID) AS TOTAL_COUNT FROM FT_INVOICE GROUP BY DEPT_ID) AS DeptInvoiceCount)"
        }
    ]
    return benchmark_data

def compare_results(generated_results, ground_truth_results):
    """Compare the results of the generated and ground truth SQL queries."""
    return set(generated_results) == set(ground_truth_results)

def evaluate_text_to_sql(llm, benchmark_data, context, difficulty):
    """Evaluate the Text-to-SQL QLLM."""
    correct_results = 0
    total_queries = len(benchmark_data)
    query_times = []
    bleu_scores = []
    valid_benchmark_queries = []

    # Difficulty and Query Clarification
    print(f"Benchmarking {total_queries} number of queries for {difficulty} difficulty")

    # Connect to the real database and Text-to-SQL QLLM
    conn = DatabaseConfiguration.connect_to_db()
    cursor = conn.cursor()

    for data in benchmark_data:
        question = data['question']
        ground_truth_sql = data['sql']

        try:
            # Checks if ground truth SQL can be executed
            cursor.execute(ground_truth_sql)
            ground_truth_results = cursor.fetchall()
            valid_benchmark_queries.append(data)
        except Exception as e:
            print(f"Error executing ground truth SQL for question: {question}\nError: {e}")
            continue

    for data in valid_benchmark_queries:
        question = data['question']
        ground_truth_sql = data['sql']

        # Default variable reset
        correct_result = False
        bleu_score = 0
        start_time = time.time()

        print(f"\n\nQuestion: {question}")
        
        try:
            # Generate SQL query using the QLLM
            generated_sql = LLMConfiguration.generate_sql_query(llm, context, question)

            # Time logging
            end_time = time.time()
            query_time = end_time - start_time
            query_times.append(query_time)
        except Exception as e:
            print(f"Error with SQL generation: {e}")
            continue

        try:
            # Generates and compares results for Generated and Grouth Truth SQL
            cursor.execute(generated_sql)
            generated_results = cursor.fetchall()
            cursor.execute(ground_truth_sql)
            ground_truth_results = cursor.fetchall()

            # SQL results matches
            if compare_results(generated_results, ground_truth_results):
                print(f"Results matched for question: {question}")
                print(f"Generated SQL: {generated_sql}")
                print(f"Ground Truth SQL: {ground_truth_sql}")
                correct_results += 1
                correct_result = True
            # SQL results do not match
            else:
                print(f"Results MISMATCHED for question: {question}")
                print(f"Generated SQL: {generated_sql}")
                print(f"Ground Truth SQL: {ground_truth_sql}")

            # Evaluate based on query similarity
            generated_tokens = nltk.word_tokenize(generated_sql)
            ground_truth_tokens = nltk.word_tokenize(ground_truth_sql)
            smoothing_function = SmoothingFunction().method4
            bleu_score = sentence_bleu([ground_truth_tokens], generated_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)

        except Exception as e:
            print(f"Failed executing generated query for question: {question}\nError: {e}")
            print(f"Generated SQL: {generated_sql}")
            print(f"Ground Truth SQL: {ground_truth_sql}")
        
        # Store SQL stat for test case to "sql_benchmark_stats"
        DatabaseConfiguration.store_sql_stat(question, correct_result, bleu_score, difficulty)
        print("Added SQL stat to sql_benchmark_stats")

    # Calculate a summary of metrics for SQL difficulty
    accuracy = correct_results / total_queries
    average_latency = sum(query_times) / total_queries
    average_bleu_score = sum(bleu_scores) / total_queries

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Latency: {average_latency:.2f} seconds")
    print(f"Average BLEU Score: {average_bleu_score:.2f}")

    # Store SQL summary to "sql_summary_stats"
    DatabaseConfiguration.store_sql_summary(difficulty, accuracy, average_bleu_score, average_latency)
    print("Added SQL summary stat to sql_summary_stats")

    # Close the database connection
    conn.close()

def run_sql_benchmark():
    # Deploy the Text-to-SQL QLLM
    nsql_llama = LLMConfiguration.deploy_nsql_llama()
    # Define the context (schema) for the test
    context = DatabaseConfiguration.get_db_schema()

    # Load benchmark dataset
    easy_benchmark_data = load_easy_benchmark_dataset()
    medium_benchmark_data = load_medium_benchmark_dataset()
    hard_benchmark_data = load_hard_benchmark_dataset()

    # Evaluate the Text-to-SQL QLLM
    evaluate_text_to_sql(nsql_llama, easy_benchmark_data, context, "easy")
    evaluate_text_to_sql(nsql_llama, medium_benchmark_data, context, "medium")
    evaluate_text_to_sql(nsql_llama, hard_benchmark_data, context, "hard")