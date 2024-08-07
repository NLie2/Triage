import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 

from datetime import datetime
import os
from dotenv import load_dotenv
import argparse

# Import key from .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


def get_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
  

def test_model_on_benchmark( model_name, nr_runs = 1): 
  """
  Run triage experiments with specified conditions
  
  model_name = specify model to use from HF 
  
  nr_runs = How many times a model will be run per condition
  
  """

  model, tokenizer =  get_model(model_name)
    
  # Load files
  df = pd.read_csv("datasets/triage_training_questions/triage_questions.csv")

  triage_scenario = json.load(open('gen_dataset/prompts/triage_context.json'))["scenario"]
  formatting_prompts = json.load(open('gen_dataset/prompts/formatting_prompts.json'))

  ethics_prompts = json.load(open('gen_dataset/prompts/ethics_prompts.json'))
  jailbreaking_prompts = json.load(open('gen_dataset/prompts/jailbreaking_prompts.json'))
  triage_syntax_variations = json.load(open('gen_dataset/prompts/triage_syntax_variations.json'))
  
  json_prompt = formatting_prompts['json']

  # model asnwers questions
  for n in range(nr_runs):
    for syntax_variation, syntax_text in triage_syntax_variations.items(): 

      for ethics_or_jb_prompt, syntax_variation in ethics_prompts.items(): 
        df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = None
        df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = None
        
        for index, row in df.iterrows():
          question_prompt = syntax_variation + row['question'] + syntax_variation + json_prompt
          print(triage_scenario)
          print(question_prompt)
          print()
          
          device = next(model.parameters()).device

          try:
            input_ids = tokenizer.encode(triage_scenario + question_prompt, return_tensors='pt', truncation=True).to(device)

            # also add model.generate and save the output 
            # set temperature to 0
            output_sequence = model.generate(input_ids, max_new_tokens=200)

            # translate output to words 
            json_response = tokenizer.decode(output_sequence, skip_special_tokens=True)


            df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = json_response["action"]
            df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = json_response["reasoning"]
            
            df.to_csv(f'triage_experiments/datasets/results/triage_results.csv')

            break

          except:
            print("an exception occured")
            continue
            break
        break
      break
    break

  print("Done!")
  print(df.head(5))


  ## Save results in file maked by date and time
  # Get date and time
  now = datetime.now().strftime("%Y-%m-%d_%H_%M")
  df.to_csv(f'triage_experiments/datasets/results/{model_name}_{now}_triage_results.csv')