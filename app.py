# libraries
import streamlit as st
import db_config
import pandas as pd
import llm_config

# # global variables
# db_config.drop_db()
# db_conn = db_config.connect_to_db()

# # setting up sqlite3 database
# books_df = pd.read_csv('datasets/books_data.csv')
# db_config.create_table('TBL_Books', books_df)
# test_context = db_config.get_db_schema()
# question = "Which are the books which were published before the year 1875?"
# nsql_llm = llm_config.deploy_nsql_llama()
# generated_query = llm_config.generate_sql_query(nsql_llm, test_context, question)
# print(generated_query)

# db_cursor = db_conn.cursor()
# db_cursor.execute(generated_query)
# rows = db_cursor.fetchall()
# for row in rows:
#         print(row)

# Streamlit Application Code
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="PromptSQL", page_icon=":robot_face:")
st.title(":chart_with_upwards_trend: PromptSQL")
st.markdown("Using local language models to power information retrieval")

# Streamlit Expander
with st.expander("Try asking me these queries"):
      st.write("""
        \n > 1) Something 
        \n > 2) Something
        \n > 3) Something
        """)      

# Streamlit Sidebar Menu
st.sidebar.caption(''':robot_face: Hello, I am a query assistant for your data. ''')
st.sidebar.button("New Chat")

# Streamlit Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:    
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

# if "messages" not in st.session_state.keys():
#       st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]