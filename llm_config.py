# libraries
import time
from langchain_community.llms import CTransformers
from accelerate import Accelerator

# global variables
accelerator = Accelerator()

# llm function calls
def deploy_nsql_llama():
    """Deploy a local instance of nsql llama 2 7b model"""
    config = {
        'max_new_tokens': 128, 
        'repetition_penalty': 1.1, 
        'context_length': 256, 
        'temperature':0, 
        'gpu_layers':50
    }

    llm = CTransformers(
        model="model/nsql-llama-2-7b.Q5_K_M.gguf",
        model_type="llama 2",
        gpu_layers = 50,
        config = config
    )
    # accelerator allows us to load configurations properly for GPU acceleration
    llm, config = accelerator.prepare(llm, config)
    return llm

def format_prompt(context, question):
    """Formats the prompt with context and question."""
    return f"{context}\n\n-- Using valid SQLite, answer the following questions for the tables provided above.\n\n-- {question}\n\nSELECT"

def generate_sql_query(llm, context, question):
    """Returns sqlite query, given the database context and question"""
    start_time = time.time()
    # prompt formatting
    # context_with_question = str(context) + "\n \n-- Using valid SQLite, answer the following questions for the tables provided above.\n\n-- " + str(question) + "\n \nSELECT"
    context_with_question = format_prompt(context, question)
    print("Prompted Question with Context: ", context_with_question)
    # feed prompt into llm for sql generation
    try:
        sql_query = "SELECT" + str(llm.invoke(context_with_question))
        # time performance console print
        end_time = time.time()
        print("Elapsed SQL generation time: ", (end_time-start_time))
        return sql_query
    except Exception as e:
        print(f"Error with query generation: {e}")

def get_llama_response():
    llm = CTransformers(model="model/llama-2-13b.Q5_K_M.gguf",
        model_type="llama 2",
        gpu_layers=100
    )