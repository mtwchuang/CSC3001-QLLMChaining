# libraries
import streamlit as st
import db_config
import pandas as pd
import llm_config
import matplotlib.pyplot as plt

# global variables
# db_config.drop_db()
# db_conn = db_config.connect_to_db()

# setting up sqlite3 database
# books_df = pd.read_csv('raw_datasets/books_data.csv')
# db_config.create_table('TBL_Books', books_df)
test_context = db_config.get_db_schema()

# Streamlit Application Code
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="PromptSQL", page_icon=":robot_face:")
st.title(":chart_with_upwards_trend: PromptSQL")
st.markdown("Using local language models to power information retrieval")

# Streamlit Expander
with st.expander("Here are some working queries that you can try asking!"):
    st.write("""
    For Invoice Dataset
    \n > Get the top three employees names and the number of times they appeared in the invoice table
    \n > Get me the top vendor names and their corresponding total payment amounts
             
    For Books Dataset
    \n > Which are the top 3 books which has the most sales in millions? Also give me their sales amounts.
    \n > Give me the top 3 authors from the database and the number of books they published
    \n > Give me the top 10 authors and their sales in millions from selling books
    """)      

# Streamlit Sidebar Menu
st.sidebar.caption(''':robot_face: Hello, I am a query assistant for your data. ''')

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

    with st.chat_message("assistant"):
        # Use nsql_llama model to generate SQL query with user question, display query
        try:
            nsql_llm = llm_config.deploy_nsql_llama()
            generated_sql_query = llm_config.generate_sql_query(nsql_llm, test_context, user_question)
            nsql_llm = None
            # removed session_state message appending for SQL query to reduce message clutter, temporary printing
            st.write(generated_sql_query)
        except Exception as error:
            print("Error: ", error)
        # Use generated SQL query to fetch data from SQLite3 database, display raw data
        try:
            db_conn = db_config.connect_to_db()
            db_curs = db_conn.cursor()
            db_curs.execute(generated_sql_query)
            raw_data = db_curs.fetchall()
            # removed session_state message appending for SQL query to reduce message clutter, temporary printing
            st.write(raw_data)
        except Exception as error:
            print("Error: ", error)

        try:
            chat_llm = llm_config.deploy_chat_llama()
            if(raw_data):
                generated_textual_insights = llm_config.generate_textual_insights(chat_llm, user_question, raw_data)
                chat_llm = None
            else:
                generated_textual_insights = "There is no results from the database that answers your question. Try rephrasing your question?"
            generated_textual_insights_str = str(generated_textual_insights)
            st.write(generated_textual_insights_str)
            st.session_state.messages.append({"role": "assistant", "content": generated_textual_insights_str})
        except Exception as error:
            print("Error: ", error)

        try:
            python_llm = llm_config.deploy_python_llama()
            if(raw_data):
                plot_code = llm_config.generate_plot_code(python_llm, user_question, raw_data)
                python_llm = None
                st.pyplot(exec(plot_code))
        except Exception as error:
            print("Error: ", error)
    
    