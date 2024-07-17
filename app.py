# libraries
import streamlit as st
import DatabaseConfiguration
import pandas as pd
import LLMConfiguration
import matplotlib.pyplot as plt
import numpy as np

# global variables
# db_config.drop_db()
# db_conn = db_config.connect_to_db()

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
    # Streamlit Expander
    with st.expander("About the Data"):
        st.write("""
        The data used in this project was synthetically generated to mimic an organization's invoice data for safe experimentation and analysis.
        
        - **FT_INVOICE (Fact Table)**: Contains primary quantitative data for analysis.
            - Columns: InvoiceID, PaymentAmount, DeptID, ProductID, VendorID, and EmployeeID
        - **LU_EMPLOYEE (Lookup Table)**: Contains information about employees who filed invoices.
            - Columns: EmployeeID, EmployeeName
        - **LU_PRODUCT (Lookup Table)**: Contains information about products listed in the invoices.
            - Columns: ProductID, ProductName
        - **LU_VENDOR (Lookup Table)**: Contains information about vendors the organization bought from.
            - Columns: VendorID, VendorFullName, CountryCode
        - **LU_DEPARTMENT (Lookup Table)**: Contains information about departments referenced
            - Columns: DeptID, DeptName
        """)
    with st.expander("Sample Queries to Try!"):
        st.write("""
        - Get the top three employees names and the number of times they appeared in the invoice table
        - Get me the top vendor names and their corresponding total payment amounts
        - Give me the top three products that are sold the most along with the number of times they have been sold.
        - Give me the top three departments who used the most total payment amounts, and their corresponding total payment amounts
        """)
    st.sidebar.caption("Options on how you want query your data:")
    show_sql_state = st.sidebar.checkbox("Show Generation SQL", True)
    show_rawdata_state = st.sidebar.checkbox("Show Raw Data Retrieved")
    show_visualisation_state = st.sidebar.checkbox("Show Visualisations")      

    # Initialize chat messages if not already present
    if "messages" not in st.session_state or st.sidebar.button("New Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input
    user_question = st.chat_input(placeholder="Ask me anything!")


    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Display the user's input correctly
        st.chat_message("user").write(user_question)
        # Boolean Flag to check if SQL query worked before continuing subsequent steps
        sucessful_sql_execution = False 

        with st.chat_message("assistant"):
            # Use nsql_llama model to generate SQL query with user question, display query
            try:
                nsql_llm = LLMConfiguration.deploy_nsql_llama()
                with st.spinner('Generating SQL Query...'):
                    generated_sql_query = LLMConfiguration.generate_sql_query(nsql_llm, st.session_state.database_context, user_question)
                    nsql_llm = None
                st.success('SQL Query Done!')
                # removed session_state message appending for SQL query to reduce message clutter, temporary printing
                if show_sql_state:
                    st.write(generated_sql_query)

            except Exception as error:
                st.error("Error in Generating SQL Query with Error Message: ", error)
                print("Error: ", error)
            # Use generated SQL query to fetch data from SQLite3 database, display raw data
            try:
                with st.spinner('Retrieving Database Information...'):
                    db_conn = DatabaseConfiguration.connect_to_db()
                    db_curs = db_conn.cursor()
                    # Verify data integrity after executing the query
                    compromised_data, table_summaries = DatabaseConfiguration.verify_data_integrity(db_curs, generated_sql_query)
                    for table, c_data in compromised_data.items():
                        if c_data:
                            DatabaseConfiguration.print_first_few_rows(c_data, label=f"Compromised Data for {table}")
                    table_summary_str = DatabaseConfiguration.get_table_summaries(table_summaries)
                    st.write(table_summary_str)
                    # db_config.print_table_summaries(table_summaries)

                    db_curs.execute(generated_sql_query)
                    raw_data = db_curs.fetchall()
                st.success('Data Retrieved!')
                # removed session_state message appending for SQL query to reduce message clutter, temporary printing
                if show_rawdata_state:
                    st.write(raw_data)
                # Indicate that data was successfully retrieved and subsequent steps can be executed
                sucessful_sql_execution = True
            except Exception as error:
                st.error("Error in Retrieving Data from Database Error Message: ", error)
                print("Error: ", error)

            if sucessful_sql_execution: 
                try:
                    chat_llm = LLMConfiguration.deploy_chat_llama()
                    with st.spinner('Generating Textual Summary...'):
                        if(raw_data):
                            generated_textual_insights = LLMConfiguration.generate_textual_insights(chat_llm, user_question, raw_data)
                            chat_llm = None
                            st.success("Textual Summary Generated!")
                        else:
                            st.warning('Your SQL query results in no results being retrieved, try rephrasing your question?', icon="⚠️")
                            generated_textual_insights = "No results retrieved"
                    generated_textual_insights_str = str(generated_textual_insights)
                    st.write(generated_textual_insights_str)
                    st.session_state.messages.append({"role": "assistant", "content": generated_textual_insights_str})
                except Exception as error:
                    st.error("Error in Generating Textual Summary with Error Message: ", error)
                    print("Error: ", error)

                if show_visualisation_state:
                    try:
                        with st.spinner('Generating Visualisations...'):
                            python_llm = LLMConfiguration.deploy_python_llama()
                            if(raw_data):
                                plot_code = LLMConfiguration.generate_plot_code(python_llm, user_question, raw_data)
                                python_llm = None
                                st.pyplot(exec(plot_code))
                        st.success("Visualisation Generated!")
                    except Exception as error:
                        st.error("Error in Generating Visualisation with Error Message: ", error)
                        print("Error: ", error)
            else:
                st.error("Subsequent steps aborted due to issues in retrieving data, try another query")

# Benchmarking Page Code
elif page == "Benchmarking Statistics":
    # Define the order of difficulties
    difficulty_order = ["easy", "medium", "hard"]
    ############ TEXT TO SQL ############
    st.header("Text-to-SQL Benchmarked Statistics")
    st.markdown("Benchmarking of the nsql-llama-2-7b.Q5_K_M model.")
    sql_summary_df = DatabaseConfiguration.read_sql_summary()
    # Sort the dataframe by difficulty order
    sql_summary_df['difficulty'] = pd.Categorical(sql_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    sql_summary_df = sql_summary_df.sort_values('difficulty')
    # Drop the 'id' column and reset the index
    sql_summary_df = sql_summary_df.drop(columns=['id']).reset_index(drop=True)
    
    # Plotting the SQL Benchmarking Data
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Accuracy vs Difficulty (Line Graph)
    ax1.plot(sql_summary_df['difficulty'], sql_summary_df['accuracy'], marker='o', linestyle='-', color='b')
    ax1.set_title("Accuracy vs Difficulty")
    ax1.set_xlabel("Difficulty")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average BLEU Score vs Difficulty (Line Graph)
    ax2.plot(sql_summary_df['difficulty'], sql_summary_df['average_bleu_score'], marker='o', linestyle='-', color='g')
    ax2.set_title("Average BLEU Score vs Difficulty")
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Average BLEU Score")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average Latency vs Difficulty
    sql_summary_df.plot(kind="barh", x="difficulty", y="average_latency", ax=ax3, legend=False)
    ax3.set_title("Average Latency vs Difficulty")
    ax3.set_xlabel("Average Latency (seconds)")
    ax3.set_ylabel("Difficulty")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)  # Close the figure to prevent overlap

    # Dataframe print
    st.dataframe(sql_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for SQL Summary Data
    sql_overall_df = sql_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(sql_overall_df, hide_index=True, use_container_width=True)

    ############ TEXT TO CHAT ############
    st.header("Text-to-Chat Benchmarked Statistics")
    st.markdown("Benchmarking of the llama-2-7b.Q5_K_M model.")
    chat_summary_df = DatabaseConfiguration.read_chat_summary()
    # Sort the dataframe by difficulty order
    chat_summary_df['difficulty'] = pd.Categorical(chat_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    chat_summary_df = chat_summary_df.sort_values('difficulty')
    # Drop the 'id' column and reset the index
    chat_summary_df = chat_summary_df.drop(columns=['id']).reset_index(drop=True)
    
    # Plotting the CHAT Benchmarking Data
    fig4, (ax6, ax7, ax8) = plt.subplots(3, 1, figsize=(12, 12))

    # Accuracy vs Difficulty (Line Graph)
    ax6.plot(chat_summary_df['difficulty'], chat_summary_df['average_bert_score'], marker='o', linestyle='-', color='b')
    ax6.set_title("BERT vs Difficulty")
    ax6.set_xlabel("Difficulty")
    ax6.set_ylabel("BERT")
    ax6.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average BLEU Score vs Difficulty (Line Graph)
    ax7.plot(chat_summary_df['difficulty'], chat_summary_df['average_rouge_score'], marker='o', linestyle='-', color='g')
    ax7.set_title("Average Rouge Score vs Difficulty")
    ax7.set_xlabel("Difficulty")
    ax7.set_ylabel("Average Rouge Score")
    ax7.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average Latency vs Difficulty
    chat_summary_df.plot(kind="barh", x="difficulty", y="average_latency", ax=ax8, legend=False)
    ax8.set_title("Average Latency vs Difficulty")
    ax8.set_xlabel("Average Latency (seconds)")
    ax8.set_ylabel("Difficulty")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)  # Close the figure to prevent overlap

    # Dataframe print
    st.dataframe(chat_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for Chat Summary Data
    chat_overall_df = chat_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(chat_overall_df, hide_index=True, use_container_width=True)

    ############ TEXT TO VIS ############
    st.header("Text-to-Vis Benchmarked Statistics")
    st.markdown("Benchmarking of the codellama-7b-python.Q5_K_M model.")
    vis_summary_df = DatabaseConfiguration.read_vis_summary()
    # Sort the dataframe by difficulty order
    vis_summary_df['difficulty'] = pd.Categorical(vis_summary_df['difficulty'], categories=difficulty_order, ordered=True)
    vis_summary_df = vis_summary_df.sort_values('difficulty')
    # Drop the 'id' column and reset the index
    # vis_summary_df = vis_summary_df.drop(columns=['id']).reset_index(drop=True)

    # Plotting Visualization Benchmark Summary Stats
    fig2, (ax4, ax9, ax10) = plt.subplots(3, 1, figsize=(12, 12))

    # vis_summary_df.plot(kind="barh", y=["average_bleu_score", "average_ssim_index", "executable_rate"], ax=ax4)

    # Average SSIM vs Difficulty (Line Graph)
    ax10.plot(vis_summary_df['difficulty'], vis_summary_df['executable_rate'], marker='o', linestyle='-', color='b')
    ax10.set_title("Execution Rate vs Difficulty")
    ax10.set_xlabel("Difficulty")
    ax10.set_ylabel("Execution Rate")
    ax10.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average BLEU Score vs Difficulty (Line Graph)
    ax4.plot(vis_summary_df['difficulty'], vis_summary_df['average_bleu_score'], marker='o', linestyle='-', color='g')
    ax4.set_title("Average BLEU Score vs Difficulty")
    ax4.set_xlabel("Difficulty")
    ax4.set_ylabel("Average BLEU Score")
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Average SSIM vs Difficulty (Line Graph)
    ax9.plot(vis_summary_df['difficulty'], vis_summary_df['average_ssim_index'], marker='o', linestyle='-', color='r')
    ax9.set_title("Average SSIM score vs Difficulty")
    ax9.set_xlabel("Difficulty")
    ax9.set_ylabel("SSIM")
    ax9.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)  # Close the figure to prevent overlap

    # Customize the plot
    # ax4.set_xlim(0, 1)  # Set x-axis limit from 0 to 1
    # ax4.xaxis.set_major_locator(plt.MultipleLocator(0.1))  # Set x-axis interval to 0.1
    # ax4.set_yticklabels(vis_summary_df['difficulty'].tolist())  # Set y-axis labels
    # ax4.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
    # ax4.legend(["Average BLEU Score", "Average SSIM Index", "Executable Rate"])
    # ax4.set_ylabel('')
    # st.pyplot(fig2)
    # plt.close(fig2)  # Close the figure to prevent overlap

    # Plotting Query Time Summary Stats
    fig3, ax5 = plt.subplots(figsize=(10, 2))
    vis_summary_df.plot(kind="barh", x="difficulty", y=["average_latency"], ax=ax5)  # Customize color if needed

    # Customize the second plot
    ax5.set_yticklabels(vis_summary_df['difficulty'].tolist())  # Set y-axis labels
    ax5.set_ylabel('')    # Remove the "difficulty" label from the y-axis
    st.pyplot(fig3)
    plt.close(fig3)  # Close the figure to prevent overlap

    st.dataframe(vis_summary_df, hide_index=True, use_container_width=True)

    # Calculate averages for Vis Summary Data
    vis_overall_df = vis_summary_df.mean(numeric_only=True).to_frame().transpose()
    st.dataframe(vis_overall_df, hide_index=True, use_container_width=True)