import DatabaseConfiguration as DC
import TextToSQLBenchmarking
import TextToChatBenchmarking
import TextToVisBenchmarking

# DC.drop_benchmark_db()
# # DC.generate_invoice_data()
# DC.connect_to_benchmark_db()
# TextToVisBenchmarking.run_visualization_benchmark()
# TextToChatBenchmarking.run_chat_benchmark()
TextToSQLBenchmarking.run_sql_benchmark()