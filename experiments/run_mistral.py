from gen_dataset.test_open_source import test_model_on_benchmark, test_model_on_benchmark_new
import sys

sys.path.append('/data/nathalie_maria_kirch/Triage')


model = "mistralai/Mistral-7B-Instruct-v0.2"

file_path = test_model_on_benchmark_new(model, test = False)

print(file_path)