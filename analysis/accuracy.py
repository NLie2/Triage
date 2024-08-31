import pandas as pd
import os


from .triage_zone_mapping import class_to_color

os.chdir('/data/nathalie_maria_kirch/Triage')

def check_match(row):
    # Get the model answer for the current gold answer from the mapping
    # Use `mapping.get(row['gold_answer'], None)` to avoid KeyError if the gold_answer is not in the mapping
    expected_model_answer = class_to_color.get(row['response'])
    # Return True if the model answer matches the expected model answer, False otherwise
    return row['triage_zone'] == expected_model_answer


def melt_df(df, df_name): 
  """
  Melt dataframe to make it easier to plot
  
  df_name = either model name if results are from single model, or condition_name if results are from multiple models
  """
  
  # only rename if question_id is not present in colnames
  if 'question_id' not in df.columns:
    df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
  df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )

  # Melt the DataFrame
  melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

  # Extract information from 'column' into new columns
  melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|Mistral|Mixtral|haiku|opus)')
  melted_df['syntax'] = melted_df['column'].str.extract('(neutral|outcome|action)')
  melted_df['prompt_type'] = melted_df['column'].str.extract('(no_ethics|deontology|utilitarianism|doctor|healthcare)')
  melted_df['response_type'] = melted_df['column'].str.extract('(reasoning|answer|raw)')

  # Filter out rows without responses  
  # ! Potentially give len of dropped responses? 
  melted_df.dropna(subset=['response'])

  # Check if the model answer matches the expected model answer
  melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

  # filter melted_df for rows that contain "answer" in response_tye
  melted_df = melted_df[melted_df['response_type'] == 'answer']


  # Drop columns that are no longer needed
  melted_df = melted_df.drop(columns=['column', 'response', 'response_type'])
  
  
  path_name = f'datasets/melted/{df_name}.csv'

  melted_df.to_csv(path_name)
  
  return path_name


def analyse_triage(path):
  """
  Analyse correctness of triage dataset
  """
  df = pd.read_csv(path)
  
  print(df.head(1))

  # print summary
  summary = df.groupby(['model', 'syntax', 'prompt_type'])['correct_answer'].mean().reset_index(name='proportion_correct')

  # summary = df.groupby(['model', 'prompt_type'])['correct_answer'].agg(['mean', 'std']).reset_index()
  # summary.columns = ['model', 'prompt_type', 'syntax','proportion_correct']
  summary = summary.sort_values(by='proportion_correct', ascending=False)
  # summary.columns = ['model', 'prompt_type','syntax','proportion_correct']



  print(summary)
  return summary
