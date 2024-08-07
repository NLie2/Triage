import sys
import json
import pandas as pd

from datetime import datetime
import os
from dotenv import load_dotenv
import argparse

# Load Models 

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# Import key from .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


def load_model(): 
  ## Load Models 
  # Mistral 
  # Mixtral 
  # Llama ? 
  # Gemini ?
  pass




def test_model_on_benchmark( model_name, nr_runs = 1): 
  """
  Run triage experiments with specified conditions
  
  model_name = specify model to use from HF 
  
  nr_runs = How many times a model will be run per condition
  
  """
  
  print("load files... ")
  # Load files
  df = pd.read_csv("datasets/triage_training_questions/triage_questions.csv")

  triage_scenario = json.load(open('gen_dataset/prompts/triage_context.json'))["scenario"]
  formatting_prompts = json.load(open('gen_dataset/prompts/formatting_prompts.json'))

  ethics_prompts = json.load(open('gen_dataset/prompts/ethics_prompts.json'))
  jailbreaking_prompts = json.load(open('gen_dataset/prompts/jailbreaking_prompts.json'))
  triage_syntax_variations = json.load(open('gen_dataset/prompts/triage_syntax_variations.json'))
  
  
  ## Ethics prompts = ethics_prompts.keys()
  ## Jailbreaking prompts = jailbreaking_prompts.keys()
  ## Syntax variations
  
  json_prompt = formatting_prompts['json']

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

          messages = [
                          {"role": "system", "content": triage_scenario},
                          {"role": "user", "content": question_prompt }
                        ]
          temperature = 0

          try:
            if model == "open-mixtral-8x22b":
                completion = mistral_client.chat(
                  model=model,
                  # messages=[ChatMessage(role="user", content="What is the best French cheese?")]
                  messages = messages, 
                  temperature= temperature
                )
                completion = completion.choices[0].message.content

            elif model == "claude-3-haiku-20240307" or model == "claude-3-opus-20240229":
                completion = anthropic_client.beta.tools.messages.create(
                  model= model,
                  max_tokens=1024,
                  system = context,
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
                print(completion)
                
                try: 
                  #haiku
                  completion = json.dumps(completion.content[0].input)
                except:
                  #opus
                  completion = json.dumps(completion.content[1].input)
                print(completion)
            else:
              completion = client.chat.completions.create(
                    model= model,
                    # response_format= { "type": "json_object" },
                    messages= messages,
                    temperature= temperature
                    )
              completion = completion.choices[0].message.content

            json_response = json.loads(completion)
            print("answered!", json_response)
            print()

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