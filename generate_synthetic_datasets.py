from faker import Faker
import pandas as pd
import random

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

# Generate synthetic data for LU_EMPLOYEE
lu_employee = pd.DataFrame({
    'EMPLOYEE_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_employees)],
    'EMPLOYEE_NAME': [fake.name() for _ in range(num_unique_employees)]
})

# Generate synthetic data for LU_VENDOR
lu_vendor = pd.DataFrame({
    'VENDOR_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_vendors)],
    'VENDOR_FULLNAME': [fake.company() for _ in range(num_unique_vendors)],
    'COUNTRY_CODE': [fake.country_code() for _ in range(num_unique_vendors)]
})

# Generate synthetic data for LU_PRODUCT
lu_product = pd.DataFrame({
    'PRODUCT_ID': [fake.unique.random_number(digits=5) for _ in range(num_unique_products)],
    'PRODUCT_NAME': [fake.bs() for _ in range(num_unique_products)],
})

# Generate synthentic data for LU_DEPARTMENT
lu_department = pd.DataFrame({
    'DEPT_ID': [fake.unique.random_number(digits=3) for _ in range(num_unique_departments)],
    'DEPT_NAME': [fake.job() for _ in range(num_unique_departments)],

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

print(lu_employee)
print(lu_vendor)
print(lu_product)
print(lu_department)
print(fact_table)

# Export to CSV
fact_table.to_csv('raw_datasets/ft_invoice.csv', index=False)
lu_employee.to_csv('raw_datasets/lu_employee.csv', index=False)
lu_vendor.to_csv('raw_datasets/lu_vendor.csv', index=False)
lu_product.to_csv('raw_datasets/lu_product.csv', index=False)
lu_department.to_csv('raw_datasets/lu_department.csv', index=False)