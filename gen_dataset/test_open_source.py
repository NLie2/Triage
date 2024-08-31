import sys
import json
import pandas as pd
import torch
import regex as re
from transformers import AutoModelForCausalLM, AutoTokenizer 

from datetime import datetime
import os
from dotenv import load_dotenv
import argparse

# Import key from .env file
load_dotenv()

import json
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set the working directory to the desired location
os.chdir('/data/nathalie_maria_kirch/Triage')


HF_TOKEN = os.getenv("HF_TOKEN")

def get_pipeline(model_name: str):
    if "mixtral" in model_name.lower():
      model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
      )
    else: 
      model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
      )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

def test_model_on_benchmark_new(model_name, nr_runs=1, test=True):
    """
    Run triage experiments with specified conditions
    
    model_name = specify model to use from HF
    nr_runs = How many times a model will be run per condition
    """
    
    generator = get_pipeline(model_name)
    
    # Load files
    df = pd.read_csv("datasets/triage_training_questions/triage_questions.csv")
    formatting_prompts = json.load(open('gen_dataset/prompts/formatting_prompts.json'))
    triage_scenario = json.load(open('gen_dataset/prompts/triage_context.json'))["scenario"]
    ethics_or_jb_prompts = json.load(open('gen_dataset/prompts/ethics_and_jailbreaking_prompts.json'))
    triage_syntax_variations = json.load(open('gen_dataset/prompts/triage_syntax_variations.json'))
    json_prompt = formatting_prompts['json']
    
    # for saving results
    now = datetime.now().strftime("%Y-%m-%d_%H_%M")

    # Model answers questions
    for n in range(nr_runs):
        for syntax_variation, syntax_text in triage_syntax_variations.items(): 
            for ethics_or_jb_prompt, ethics_or_jb_text in ethics_or_jb_prompts.items(): 
                df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = None
                df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = None
                
                for index, row in df.iterrows():
                    question_prompt = syntax_text + row['question'] + ethics_or_jb_text + json_prompt
                    full_prompt = triage_scenario + question_prompt 
                    print(full_prompt)
                    print()

                    try:
                        # Generate output using the pipeline
                        response = generator(full_prompt, num_return_sequences=1, max_new_tokens=200, do_sample=False)
                        response_text = response[0]['generated_text']

                        # Remove the prompt from the start of the generated text
                        if response_text.startswith(full_prompt): response_text = response_text[len(full_prompt):].strip()
                        
                        print(response_text)
        
                        # Use regular expression to extract JSON object
                        json_matches = re.findall(r'\{.*?\}', response_text, re.DOTALL)
                        if not json_matches:
                                raise ValueError("No JSON object found in the response")
                        json_str = json_matches[-1]
                        json_response = json.loads(json_str)

                        
                        print("Parsed JSON response:", json_response)
                        
                        df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = json_response.get("action")
                        df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = json_response.get("reasoning")
                        
                        file_path = f'datasets/results/{model_name.replace('/', '_')}_{now}_triage_results_temp.csv'

                        df.to_csv(file_path)
  
                        if test:
                            break

                    except Exception as e:
                        print("An exception occurred")
                        print(e)
                        if test:
                            break
                        continue

                if test:
                    break

            if test:
                break

        if test:
            break

    print("Done!")
    print(df.head(5))
    
    ## Save results in file maked by date and time
    # Get date and time

    file_path = f'datasets/results/{model_name.replace('/', '_')}_{now}_triage_results.csv'

    df.to_csv(file_path)
    return file_path


def get_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_model_on_benchmark( model_name, nr_runs = 1, test=True): 
  """
  Run triage experiments with specified conditions
  
  model_name = specify model to use from HF 
  
  nr_runs = How many times a model will be run per condition
  
  """

  generator = get_pipeline(model_name)
    
  # Load files
  df = pd.read_csv("datasets/triage_training_questions/triage_questions.csv")
  
  formatting_prompts = json.load(open('gen_dataset/prompts/formatting_prompts.json'))

  triage_scenario = json.load(open('gen_dataset/prompts/triage_context.json'))["scenario"]

  ethics_or_jb_prompts = json.load(open('gen_dataset/prompts/ethics_and_jailbreaking_prompts.json'))
  triage_syntax_variations = json.load(open('gen_dataset/prompts/triage_syntax_variations.json'))
  
  json_prompt = formatting_prompts['json']

  # model asnwers questions
  for n in range(nr_runs):
    for syntax_variation, syntax_text in triage_syntax_variations.items(): 

      for ethics_or_jb_prompt, syntax_variation in ethics_or_jb_prompts.items(): 
        df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = None
        df[f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = None
        
        for index, row in df.iterrows():
            question_prompt = syntax_text + row['question'] + json_prompt
            print(triage_scenario)
            print(question_prompt)
            print()


          # try:

            # also add model.generate and save the output 
            # set temperature to 0
            response = generator(question_prompt, num_return_sequences=1, max_new_tokens=200, do_sample=True)
            
            print(response)
            response[0].replace("'", '"')
    
            print("original", response)
            print()
            # make sure response is string
            response = str(response)
            
            json_response = json.loads('{ "reasoning":"Because the patient... they should be assigned category ... ", "action": "MINOR" }')
            print("literally printed", json_response)
            
            json_response = json.loads(response)
            
            print("original as json", json_response)
            
            df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = json_response["action"]
            df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = json_response["reasoning"]
            
            df.to_csv(f'triage_experiments/datasets/results/triage_results.csv')

            if test: break

          # except Exception as e:
          #   print("an exception occured")
          #   print(e)          
          #   if test: break
          #   continue 

        if test: break
        

      if test: break

    if test: break

  print("Done!")
  print(df.head(5))
  
  
  ## Save results in file maked by date and time
  # Get date and time
  now = datetime.now().strftime("%Y-%m-%d_%H_%M")

  file_path = f'datasets/results/{model_name.replace('/', '_')}_{now}_triage_results.csv'

  df.to_csv(file_path)
  return file_path