import DatabaseConfiguration
import TextToSQLBenchmarking
import TextToChatBenchmarking
import TextToVisBenchmarking
from ctransformers import AutoModelForCausalLM

# Creation of Invoice Datasets
DatabaseConfiguration.drop_db()
DatabaseConfiguration.create_invoice_db()
DatabaseConfiguration.generate_invoice_data()

# Downloads a base model for its tokenizer
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_type="llama", gpu_layers=50)

# Creation of Benchmarks (Recommended to Run Only when GPU has been Accelerated, commented out until then)
# DatabaseConfiguration.drop_benchmark_db()
# DatabaseConfiguration.connect_to_benchmark_db()
# TextToSQLBenchmarking.run_sql_benchmark()
# TextToChatBenchmarking.run_chat_benchmark()
# TextToVisBenchmarking.run_visualization_benchmark()