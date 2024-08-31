from gen_dataset.test_open_source import test_model_on_benchmark, test_model_on_benchmark_new
import sys

sys.path.append('/data/nathalie_maria_kirch/Triage')


model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

file_path = test_model_on_benchmark_new(model, test = False)

print(file_path)