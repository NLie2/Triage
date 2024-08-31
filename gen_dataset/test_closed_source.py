import sys
import json
import pandas as pd
import torch
import regex as re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 

from datetime import datetime
import os
from dotenv import load_dotenv
import argparse

from openai import OpenAI
from anthropic import Anthropic

# Import key from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# Set the working directory to the desired location
os.chdir('/data/nathalie_maria_kirch/Triage')

def test_model_on_benchmark_closed(model_name, nr_runs=1, test=True):
    """
    Run triage experiments with specified conditions
    
    model_name = specify model to use from API
    nr_runs = How many times a model will be run per condition
    """
        
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
                    
                    messages = [
                      {"role": "system", "content": triage_scenario},
                      {"role": "user", "content": question_prompt }
                      ]
                    temperature = 0

                    try:
                      if model_name == "claude-3-haiku-20240307" or model_name == "claude-3-opus-20240229":
                        completion = anthropic_client.messages.create(
                          model= model_name,
                          max_tokens=1024,
                          temperature = temperature, 
                          system = triage_scenario,
                          messages=[
                            {"role": "user", "content": question_prompt }
                            ],
                          tools=[
                            {
                              "name": "triage_classification",
                              "description": "A JSON object containing decision and reasoning about which category will be assigned to a person and why.",
                              "input_schema": {
                                "type": "object",
                                "properties": {
                                  "reasoning": {
                                    "type": "string",
                                    "description": "The reasoning behind the decision."
                                    },
                                  "action": {
                                    "type": "string",
                                    "description": "The category 'MINOR', 'EXPECTANT/DECEASED', 'DELAYED', or 'IMMEDIATE'."
                                    }
                                  }
                                }
                              } 
                            ]
                          ) 
                        try: 
                          #haiku
                          completion = json.dumps(completion.content[0].input)
                        except:
                          #opus
                          completion = json.dumps(completion.content[1].input)
                        
                        json_response = json.loads(completion)
                        print(completion)
                        print(json_response)
                                             
                        print("Parsed JSON response:", json_response)
                        
                        df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_answer_{n}'] = json_response["action"]
                        df.loc[index, f'{model_name}_{syntax_variation}_{ethics_or_jb_prompt}_reasoning_{n}'] = json_response["reasoning"]
                        
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
