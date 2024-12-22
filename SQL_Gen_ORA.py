import pandas as pd
import os
from openai import OpenAI

from google.colab import drive
drive.mount('/content/drive')

# Creating base directories
base_dir = '/content/drive/MyDrive/RTS_Chetan_Chathurvedhi'

## Client

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

model_table = {
    "phi": "microsoft/phi-3-medium-128k-instruct:free",
    "lfm":"liquid/lfm-40b:free",
  }

# Response function to generate sql query

counter = 0

def generate_sql_query(model, natural_language_query, schema, top_entries, shot):
  if shot == 0:
    prompt = f"Write an SQL query to fulfill the following natural language request:\n\n" \
             f"Request: {natural_language_query}\n"
  elif shot == 1:
    prompt = f"Write an SQL query to fulfill the following natural language request:\n\n" \
             f"Request: {natural_language_query}\n" \
             f"Schema of the tables: {schema}\n"
  else:
    prompt = f"Write an SQL query to fulfill the following natural language request:\n\n" \
             f"Request: {natural_language_query}\n" \
             f"Schema of the tables: {schema}\n" \
             f"Top entries of the table:\n{top_entries}\n"

  completion = client.chat.completions.create(
      model = model_table[model],
      messages=[
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries, provide only the code, no other comments"},
            {"role": "user", "content": prompt}
      ]
  )


  return completion.choices[0].message.content

df_simple = pd.read_csv(base_dir+'/Datasets/crime_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(base_dir+'/Datasets/crime_complex.csv',encoding='windows-1254')

anal_loc = base_dir + '/Prompt_Analysis'
model = 'phi'

# Zero Shot

df_simple['Zero_Shot'] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], '', '', 0), axis=1)
df_complex['Zero_Shot'] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], '', '', 0), axis=1)

# One Shot

df_simple['One_Shot'] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], '', 1), axis=1)
df_complex['One_Shot'] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], '', 1), axis=1)

# Few Shot

df_simple['Few_Shot'] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
df_complex['Few_Shot'] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)

# Saving the file

df_simple.to_csv(anal_loc+'/crime_simple.csv', index=False)
df_complex.to_csv(anal_loc+'/crime_complex.csv', index=False)

# Crime Dataset

df_simple = pd.read_csv(base_dir+'/Datasets/crime_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(base_dir+'/Datasets/crime_complex.csv',encoding='windows-1254')

res_loc = base_dir + '/Results/Crime'

crime_models = []

for model in model_table:
  print(model, end = ' ')
  if(model in crime_models):
    print('Done')
    continue
  df_simple[model] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_complex[model] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_simple.to_csv(res_loc+'/crime_simple.csv', index=False)
  df_complex.to_csv(res_loc+'/crime_complex.csv', index=False)
  crime_models.append(model)
  print('Evaluated')
  
# Shooting Dataset
  
df_simple = pd.read_csv(base_dir+'/Datasets/shooting_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(base_dir+'/Datasets/shooting_complex.csv',encoding='windows-1254')

res_loc = base_dir + '/Results/Shooting'

shooting_models = []

for model in model_table:
  print(model, end = ' ')
  if(model in shooting_models):
    print('Done')
    continue
  df_simple[model] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_complex[model] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_simple.to_csv(res_loc+'/shooting_simple.csv', index=False)
  df_complex.to_csv(res_loc+'/shooting_complex.csv', index=False)
  shooting_models.append(model)
  print('Evaluated')
  
# Housing Dataset
  
df_simple = pd.read_csv(base_dir+'/Datasets/housing_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(base_dir+'/Datasets/housing_complex.csv',encoding='windows-1254')

res_loc = base_dir + '/Results/Housing'

housing_models = []

for model in model_table:
  print(model, end = ' ')
  if(model in housing_models):
    print('Done')
    continue
  df_simple[model] = df_simple.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_complex[model] = df_complex.apply(lambda row: generate_sql_query(model, row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)
  df_simple.to_csv(res_loc+'/housing_simple.csv', index=False)
  df_complex.to_csv(res_loc+'/housing_complex.csv', index=False)
  housing_models.append(model)
  print('Evaluated')

