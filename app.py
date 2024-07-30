# libraries
import streamlit as st
import DatabaseConfiguration
import pandas as pd
import LLMConfiguration
import matplotlib.pyplot as plt
import time

# Initialize database if not already done
if "db_initialized" not in st.session_state:
    DatabaseConfiguration.create_invoice_db()
    st.session_state.database_context = DatabaseConfiguration.get_db_schema()
    # DatabaseConfiguration.connect_to_benchmark_db()
    st.session_state.db_initialized = True

# Streamlit Application Code
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="QLLMChain", page_icon=":robot_face:")
st.title(":chart_with_upwards_trend: QLLMChain")
st.markdown("Using quantised locally deployed large language models to power information retrieval and intepretation")

# Streamlit Sidebar Menu
st.sidebar.caption(''':robot_face: Hello, I am a query assistant for your data. ''')
page = st.sidebar.selectbox("Select a page:", ["Main", "Benchmarking Statistics"])

# Main Page Code
if page == "Main":
    # About data expander
    with st.expander("About the Data"):
        st.write("""
Data used in this project was synthetically generated to mimic invoice data for an organisation.
- **FT_INVOICE (Fact Table)**: Contains primary quantitative data for analysis.
    
    Columns: InvoiceID, PaymentAmount, DeptID, ProductID, VendorID, and EmployeeID
- **LU_EMPLOYEE (Lookup Table)**: Contains information about employees who filed invoices.

    Columns: EmployeeID, EmployeeName
- **LU_PRODUCT (Lookup Table)**: Contains information about products listed in the invoices.
                 
    Columns: ProductID, ProductName
- **LU_VENDOR (Lookup Table)**: Contains information about vendors the organization bought from.
                 
    Columns: VendorID, VendorFullName, CountryCode
- **LU_DEPARTMENT (Lookup Table)**: Contains information about departments referenced
    
    Columns: DeptID, DeptName
        """)
    # Sample queries expander
    with st.expander("Sample Queries to Try!"):
        st.write("""
Data Exploration (No Visualisation):
1. Show me the all the different departments from the department table
2. Show me all the different vendors from the vendor table
3. Show me all the different employees from the employee table
4. Show me all the different product types from the product table

Medium Queries (Top Three Retrievals with Joins):
1. Get the top three employees names and the number of times they appeared in the invoice table?
2. Which are the top three departments who spent the most total payment amount and their corresponding total payment amounts?
3. Which are the top three vendors who been paid the most total payment amount and their corresponding total payments amounts earned?
4. Which are the top three products that has been spent the most total payment amount on and their corresponding total payment amounts?
                 
Hard Queries (With joins and conditional filtering)
1. How much total payment amount did the IT department spend on the Product named, "Product 7"?
2. How much total payment amount did the employee named "Taylor Walker" handle through all the invoices? 
        """)
# - Get me the top vendor names and their corresponding total payment amounts
# - Give me the top three products that are sold the most along with the number of times they have been sold.
# - Give me the top three departments who used the most total payment amounts, and their corresponding total payment amounts

    # Page Variables
    st.sidebar.caption("Options on how you want query your data:")
    show_sql_state = st.sidebar.checkbox("Show Generated SQL", True)
    show_rawdata_state = st.sidebar.checkbox("Show Raw Data Retrieved")
    show_visualisation_state = st.sidebar.checkbox("Show Visualisations")
    st.sidebar.caption("For testing error handling purposes:")
    negate_sql_execution_state = st.sidebar.checkbox("Negate SQL Execution State")
    negate_presence_of_data_state =  st.sidebar.checkbox("Negate Data Presence State")
    negate_data_integrity_state =  st.sidebar.checkbox("Negate Data Integrity State")

    # Initialize chat messages if not already present
    if "messages" not in st.session_state or st.sidebar.button("New Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input
    user_question = st.chat_input(placeholder="Ask me anything!")

    if user_question:
        # Create the user icon and display user question accordingly
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        print(f"""\n\n========================================================================================\n=========== New User Query : {user_question} ===========\n========================================================================================""")

        # Boolean Flags
        successful_sql_execution = False     # SQL Execution State
        data_integrity = False               # Data Integrity State
        retrieved_data_presence = False      # Non-Empty Data State

        with st.chat_message("assistant"):
            try:
                # Deploy and Generate SQL Query using NSQL-Llama-2-7b-Q5-K-M
                nsql_llm = LLMConfiguration.deploy_nsql_llama()
                with st.spinner('Generating SQL Query...'):
                    sql_start_time = time.time()
                    generated_sql_query = LLMConfiguration.generate_sql_query(nsql_llm, st.session_state.database_context, user_question)
                    nsql_llm = None
                    # SQL Time Elapsed
                    sql_end_time = time.time()
                    sql_elapsed_time = sql_end_time - sql_start_time
                    # Displays SQL query if true
                    if show_sql_state:
                        st.write(generated_sql_query)
                st.success(f'SQL query successfully generated in {sql_elapsed_time:.2f}s!')
            # SQL Generation Exception Handling
            except Exception as error:
                st.error("Error in SQL query generation, check console logs for errors and try another query.")
                print("Error: ", error)

            try:
                # Fetch data from SQLite3 Database
                with st.spinner('Retrieving Database Information...'):
                    db_conn = DatabaseConfiguration.connect_to_db()
                    db_curs = db_conn.cursor()
                    db_start_time = time.time()
                    db_curs.execute(generated_sql_query)
                    raw_data = db_curs.fetchall()
                    # Get column names involved
                    column_names = [description[0] for description in db_curs.description]
                    print(f'Column names retrieved: {column_names}')
                    # DB Time Elapsed
                    db_end_time = time.time()
                    db_elapsed_time = db_end_time - db_start_time
                    # Displays retrieved data if true
                    if show_rawdata_state:
                        st.write(raw_data)
                    # Flag set as true if there is data
                    if raw_data:
                        retrieved_data_presence = True
                    # Flag set as true for successful data retrieval
                    successful_sql_execution = True
                st.success(f'Data retrieved from database in {db_elapsed_time:.2f}s!')
            # Data Retrieval Exception Handling
            except Exception as error:
                st.error("Error in data retrieval, check console logs for errors and try another query.")
                print("Error: ", error)

            try:
                # Data integrity checks using SHA256 Hash Comparisons for used tables in SQL Query
                with st.spinner('Performing Data Integrity Checks...'):
                    check_start_time = time.time()
                    compromised_data, table_summaries = DatabaseConfiguration.verify_data_integrity(db_curs, generated_sql_query)
                    # Prints compromised rows if any
                    for table, c_data in compromised_data.items():
                        if c_data:
                            DatabaseConfiguration.print_first_few_rows(c_data, label=f"Compromised Data for {table}")
                        else:
                            # Flag set as true for passing data integrity check
                            data_integrity = True
                    # Check time elapsed
                    check_end_time = time.time()
                    check_elapsed_time = check_end_time - check_start_time
                # Display the table summaries using Streamlit's dataframe
                table_summary_str = DatabaseConfiguration.get_table_summaries(table_summaries)
                for table_summary in table_summary_str:
                    st.markdown(f"#### Table: {table_summary['Table']}")

                    # Use columns to arrange the elements side by side
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Total Rows", value=table_summary['Total Rows'])
                    col2.metric(label="Verified Rows", value=table_summary['Verified Rows'])
                    col3.metric(label="Percentage Verified", value=f"{table_summary['Percentage Verified']}")

                    # Display the progress bar below the columns
                    percentage_verified = float(table_summary['Percentage Verified'].strip('%'))
                    st.progress(percentage_verified / 100)
                    st.markdown("---")
                st.success(f'Data integrity checks completed in {check_elapsed_time:.2f}s!')
            # Data Integrity Check Handling
            except Exception as error:
                st.error("Error in data integrity checks, please check console logs.")
                print("Error: ", error)

            # Override values according to side checkboxes
            if negate_sql_execution_state:
                successful_sql_execution = False
            if negate_data_integrity_state:
                data_integrity = False
            if negate_presence_of_data_state:
                retrieved_data_presence = False


            if successful_sql_execution and data_integrity and retrieved_data_presence:
                # If data has been retrieved and not tampered with
                try:
                    # Deploy and Generate Textual Summaries using Llama-2-7b-Q5-K-M
                    chat_llm = LLMConfiguration.deploy_chat_llama()
                    with st.spinner('Generating Textual Summary...'):
                        chat_start_time = time.time()
                        generated_textual_insights = LLMConfiguration.generate_textual_insights(chat_llm, user_question, raw_data)
                        chat_llm = None
                        # Chat time elapsed
                        chat_end_time = time.time()
                        chat_elapsed_time = chat_end_time - chat_start_time
                    # Formatting bulletpoints for proper display on Streamlit
                    formatted_textual_insights = generated_textual_insights.replace('•', '\n\n•')
                    formatted_textual_insights = formatted_textual_insights + f"\n\nNames of columns retrieved: {column_names}"
                    st.markdown(f"""{formatted_textual_insights}""")
                    st.success(f"Textual summary generated in {chat_elapsed_time:.2f}s!")
                    # Saves message to session state, for message to persist
                    st.session_state.messages.append({"role": "assistant", "content": formatted_textual_insights})
                # Textual summary error handling
                except Exception as error:
                    st.error("Error in textual summary generation, please check console logs for errors and try another query.")
                    print("Error: ", error)

                if show_visualisation_state:
                    # Perform Visualisation Generation if True
                    try:
                        # Deploy and Generate Python Code for Visualisations using CodeLlama-Python-7b-Q5-K-M
                        with st.spinner('Generating Visualisations...'):
                            python_llm = LLMConfiguration.deploy_python_llama()
                            if(raw_data):
                                vis_start_time = time.time()
                                plot_code = LLMConfiguration.generate_plot_code(python_llm, user_question, raw_data)
                                python_llm = None
                                # Visualisation time elapsed
                                vis_end_time = time.time()
                                vis_elapsed_time = vis_end_time - vis_start_time
                                # Attempting to execute the python code for visualisation
                                exec_globals = {}
                                exec(plot_code, exec_globals)
                                st.pyplot(exec_globals.get('plt'))
                        st.success(f"Visualisation generated in {vis_elapsed_time:.2f}s!")
                    except Exception as error:
                        st.error("Error in visualisation generation, please check the console logs and try another query")
                        print("Error: ", error)

            elif retrieved_data_presence and data_integrity:
                # Skips further steps, data retrieval fails
                st.error("Subsequent steps aborted due to errors in data retrieval , try another query.")
            elif data_integrity and successful_sql_execution:
                # Skips further steps, no data retrieved
                st.error("Subsequent steps aborted due to no data retrieved , try another query.")
            elif retrieved_data_presence and successful_sql_execution:
                # Skips further steps, if integrity fails
                st.error("Subsequent steps aborted due to failure in data integrity, please check the data.")
            else:
                # Skip further steps, due to failure in multiple conditions
                st.error(f"Subsequent steps aborted due to multiple condition failures, the following states are\n\n - Presence of Data: {retrieved_data_presence}\n - Successful Data Execution: {successful_sql_execution}\n - Data Integrity State: {data_integrity}")

# Benchmarking Page Code
elif page == "Benchmarking Statistics":
    # Define the order of difficulties
    difficulty_order = ["easy", "medium", "hard"]
    
    ############ TEXT TO SQL ############
    st.header("Text-to-SQL Benchmarked Statistics")
    st.markdown("Benchmarking of the nsql-llama-2-7b.Q5_K_M model.")
    sql_summary_df = DatabaseConfiguration.read_sql_summary()
    sql_summary_df['difficulty'] = pd.Categorical(sql_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    sql_summary_df = sql_summary_df.drop(columns=['id']).reset_index(drop=True)
    sql_summary_df = sql_summary_df.sort_values('difficulty').reset_index(drop=True)
    
    # Create columns for combined metrics and latency plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Combine metrics into a single bar plot, grouped by metrics
        sql_metrics_df = sql_summary_df.melt(id_vars=['difficulty'], value_vars=['accuracy', 'average_bleu_score'], var_name='Metric', value_name='Value')
        fig, ax = plt.subplots(figsize=(12, 6))
        sql_metrics_df.pivot(index='Metric', columns='difficulty', values='Value').plot(kind='bar', ax=ax)
        ax.set_title("Text-to-SQL Metrics by Difficulty")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Values")
        ax.legend(title='Difficulty')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(['Accuracy', 'Average BLEU Score'], rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    with col2:
        # Plotting latency separately
        fig, ax = plt.subplots(figsize=(12, 6))
        sql_summary_df.set_index('difficulty')[['average_latency']].plot(kind='bar', ax=ax, color='orange')
        ax.set_title("Text-to-SQL Average Latency by Difficulty")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Average Latency (seconds)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(difficulty_order, rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    st.dataframe(sql_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for SQL Summary Data
    sql_overall_df = sql_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(sql_overall_df, hide_index=True, use_container_width=True)
    st.markdown("---")
    
    ############ TEXT TO CHAT ############
    st.header("Text-to-Chat Benchmarked Statistics")
    st.markdown("Benchmarking of the llama-2-7b.Q5_K_M model.")
    chat_summary_df = DatabaseConfiguration.read_chat_summary()
    chat_summary_df['difficulty'] = pd.Categorical(chat_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    chat_summary_df = chat_summary_df.drop(columns=['id']).reset_index(drop=True)
    chat_summary_df = chat_summary_df.sort_values('difficulty').reset_index(drop=True)
    
    # Create columns for combined metrics and latency plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Combine metrics into a single bar plot, grouped by metrics
        chat_metrics_df = chat_summary_df.melt(id_vars=['difficulty'], value_vars=['average_bert_score', 'average_rouge_score'], var_name='Metric', value_name='Value')
        fig, ax = plt.subplots(figsize=(12, 6))
        chat_metrics_df.pivot(index='Metric', columns='difficulty', values='Value').plot(kind='bar', ax=ax)
        ax.set_title("Text-to-Chat Metrics by Difficulty")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Values")
        ax.legend(title='Difficulty')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(['Average BERT Score', 'Average Rouge Score'], rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    with col2:
        # Plotting latency separately
        fig, ax = plt.subplots(figsize=(12, 6))
        chat_summary_df.set_index('difficulty')[['average_latency']].plot(kind='bar', ax=ax, color='orange')
        ax.set_title("Text-to-Chat Average Latency by Difficulty")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Average Latency (seconds)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(difficulty_order, rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    st.dataframe(chat_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for Chat Summary Data
    chat_overall_df = chat_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(chat_overall_df, hide_index=True, use_container_width=True)
    st.markdown("---")
    
    ############ TEXT TO VIS ############
    st.header("Text-to-Vis Benchmarked Statistics")
    st.markdown("Benchmarking of the codellama-7b-python.Q5_K_M model.")
    vis_summary_df = DatabaseConfiguration.read_vis_summary()
    vis_summary_df['difficulty'] = pd.Categorical(vis_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    vis_summary_df = vis_summary_df.sort_values('difficulty').reset_index(drop=True)
    
    # Create columns for combined metrics and latency plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Combine metrics into a single bar plot, grouped by metrics
        vis_metrics_df = vis_summary_df.melt(id_vars=['difficulty'], value_vars=['executable_rate', 'average_ssim_index', 'average_pixel_similarity', 'average_bleu_score'], var_name='Metric', value_name='Value')
        fig, ax = plt.subplots(figsize=(12, 6))
        vis_metrics_df.pivot(index='Metric', columns='difficulty', values='Value').plot(kind='bar', ax=ax)
        ax.set_title("Text-to-Vis Metrics by Difficulty")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Values")
        ax.legend(title='Difficulty')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(['Execution Rate', 'Average SSIM Index', 'Average Pixel Similarity', 'Average BLEU Score'], rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    with col2:
        # Plotting latency separately
        fig, ax = plt.subplots(figsize=(12, 6))
        vis_summary_df.set_index('difficulty')[['average_latency']].plot(kind='bar', ax=ax, color='orange')
        ax.set_title("Text-to-Vis Average Latency by Difficulty")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Average Latency (seconds)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(difficulty_order, rotation=0)  # Custom x-axis labels
        st.pyplot(fig)
        plt.close()

    st.dataframe(vis_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for Vis Summary Data
    vis_overall_df = vis_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(vis_overall_df, hide_index=True, use_container_width=True)