# libraries
import time
from langchain_community.llms import CTransformers
from accelerate import Accelerator

# global variables
accelerator = Accelerator()

"""LLM CTransformer Deployment Functions

Configuration Parameters: 
max_new_tokens      (int) - maximum number of tokens the model can generate in a single output
reptition_penalthy  (int) - likelihood of the model repeating the same line or phrase
context_length      (int) - number of tokens in the input that the model considers for generating an output
temperature         (int) - randomness of the output generation. A lower temperature results in more predictable and conservative outputs
gpu_layers          (int) - layers of the model should be processed on the GPU

Model Initialisation Parameters:
model       - Path to model
model_type  - Name of model
gpu_layers  - layers of the model should be processed on the GPU
config      - configuration dictionary defined
"""

def deploy_nsql_llama():
    """Deploy a local instance of nsql llama 2 7b model"""
    config = {
        'max_new_tokens': 256, 
        'repetition_penalty': 1.1, 
        'context_length': 512, 
        'temperature':0, 
        'gpu_layers':50
    }

    llm = CTransformers(
        model="model/nsql-llama-2-7b.Q5_K_M.gguf",
        model_type="nsql llama 2",
        gpu_layers = 50,
        config = config
    )
    # accelerator allows us to load configurations properly for GPU acceleration
    llm, config = accelerator.prepare(llm, config)
    return llm

def deploy_chat_llama():
    """Deploy a local instance of the chat llama 2 7b model"""
    config = {
        # max tokens and context length may have to be adjusted based on amount of output coming out from DB
        'max_new_tokens': 512, 
        'repetition_penalty': 1.1, 
        'context_length': 1536, 
        'temperature':0.5, 
        'gpu_layers':50
    }

    llm = CTransformers(
        model="model/llama-2-7b-chat.Q5_K_M.gguf",
        model_type="llama 2",
        gpu_layers = 50,
        config = config
    )
    # accelerator allows us to load configurations properly for GPU acceleration
    llm, config = accelerator.prepare(llm, config)
    return llm

def deploy_python_llama():
    """Deploy a local instance of the python 2 7b model"""
    config = {
        'max_new_tokens': 1024, 
        'repetition_penalty': 1.1, 
        'context_length': 1536, 
        'temperature':0, 
        'gpu_layers':50
    }

    llm = CTransformers(
        model="model/codellama-7b-python.Q5_K_M.gguf",
        model_type="code llama python",
        gpu_layers = 50,
        config = config
    )
    # accelerator allows us to load configurations properly for GPU acceleration
    llm, config = accelerator.prepare(llm, config)
    return llm

"""Large Language Model Function Calls"""

def generate_sql_query(llm, context, question):
    """Returns sqlite query, given the database context and question"""
    start_time = time.time()
    # prompt formatting (Prompts + Context + Question)
    crafted_prompt =  str(context) + f"""
-- Using valid SQLite, answer the following questions for the tables provided above.
-- Make necessary table joins when needed, and avoid them if it is unnecessary.
-- Take note of column names belonging to each table and make sure you do not select wrong column from wrong tables.
-- Ensure that you are using the correct table alias if any are provided.
-- This is the question: {str(question)} 
SELECT"""
    print("Crafted Prompt:\n", crafted_prompt)
    # feed prompt into llm for sql generation
    try:
        cleaned_sql_query = "SELECT" + str(llm.invoke(crafted_prompt))
        print("\nSQL Query: ",cleaned_sql_query)
        # time performance console print
        end_time = time.time()
        print("Elapsed SQL generation time: ", (end_time-start_time))
        return cleaned_sql_query
    except Exception as e:
        print(f"Error with query generation: {e}")

def generate_textual_insights(llm, question, raw_data):
    """Returns textual insights based on the raw data extracted from the database"""
    max_context_length = 1024
    start_time = time.time()
    # prompt formatting (Prompts + Question + Data + Syntax)
    crafted_prompt = f"""[INST] <<SYS>> Based on the question and data given, answer the question using text with the data. No explanation is required. Be careful to look out for reptitions. Use bullet points for newlines to ensure proper formatting. Ensure all information is given is reflected. <</SYS>> 
Here is my question. "{question}". Based on the question these are the results: {raw_data}[/INST]"""
    print("Crafted Prompt:\n", crafted_prompt)
    # Calculates total tokens required for context, 1.349 is a multiplier for more accuracy
    actual_token_multiplier = 1.349
    num_of_tokens = llm.get_num_tokens(str(crafted_prompt))
    print("Number of tokens is: ", num_of_tokens*actual_token_multiplier)

    try:
        # if context does not exceed context length
        if(max_context_length > num_of_tokens*actual_token_multiplier):
            # Generation of Textual Summary by Llama QLLM
            textual_insights = str(llm.invoke(str(crafted_prompt)))
            print(textual_insights)
            # Time Logging
            end_time = time.time()
            print("Elapsed textual insight generation time: ",(end_time - start_time))
            return textual_insights
        else:
            return "Data retrieved too large, try to narrow down your search question?"

    except Exception as e:
        print(f"Error with textual insight generation: {e}")

def generate_plot_code(llm, question, raw_data):
    """Returns python code based on prompts and raw data extracted from the database"""
    start_time = time.time()
    # Prompt Engineering, supply question, data to use and specifically not to use other sources of data
    raw_data_str = str(raw_data)
    crafted_prompt = f"""[INST] Based on the question given as: {question}, and the dataframe given as {raw_data_str}, generate matplotlib code using this dataframe to answer the question. The data is already provided as a list of tuples, so there's no need to read any external data sources, including URLs or CSV files. Only use the provided dataframe for the visualization. Pay attention to the structure of the data and the types of operations that are valid for it. Pay attention to the question and use the right metric and include it in the graph.[/INST]"""
    print(crafted_prompt)

    try:
        # Generation of Matplotlib Code by Python QLLM
        python_code = llm.invoke(str(crafted_prompt))
        # removal of python magic commands
        python_code = '\n'.join([line for line in python_code.split('\n') if not line.startswith('%') and 'get_ipython()' not in line])
        print(python_code)
        # Time Log
        end_time = time.time()
        print("Elapsed plot code generation time: ",(end_time - start_time))
        return python_code
    
    except Exception as error:
        print('Error: ', error)
        return "Error with visualisation"
