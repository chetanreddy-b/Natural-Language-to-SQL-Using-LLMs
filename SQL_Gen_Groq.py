import pandas as pd
import openai
import os
from groq import Groq
import chardet
import shutil

from google.colab import drive
drive.mount('/content/drive')

# Creating base directories
base_dir = '/content/drive/MyDrive/RTS_Chetan_Chathurvedhi'

client = Groq(
    api_key="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",
)

groq_table = {
    "gemma_1":"gemma-7b-it",
    "gemma_2":"gemma2-9b-it",
    "llama":"llama3-groq-8b-8192-tool-use-preview",
    "mistral":"mixtral-8x7b-32768"

}

# Response function to generate sql query - Groq

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

  response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries."},
            {"role": "user", "content": prompt}
        ],
        model=groq_table[model]
  )

  generated_sql = response.choices[0].message.content.strip()

  return generated_sql

res_loc = base_dir + '/Results/Crime'

df_simple = pd.read_csv(res_loc+'/crime_simple.csv',encoding='utf8')
df_complex = pd.read_csv(res_loc+'/crime_complex.csv',encoding='windows-1254')

crime_models = []

for model in groq_table:
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
  
res_loc = base_dir + '/Results/Shooting'

df_simple = pd.read_csv(res_loc+'/shooting_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(res_loc+'/shooting_complex.csv',encoding='windows-1254')

shooting_models = []

for model in groq_table:
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
  
res_loc = base_dir + '/Results/Housing'

df_simple = pd.read_csv(res_loc+'/housing_simple.csv',encoding='windows-1254')
df_complex = pd.read_csv(res_loc+'/housing_complex.csv',encoding='windows-1254')

housing_models = []

for model in groq_table:
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
  
# Setting up the SQL folder
  
res_loc = base_dir + '/Results'
sql_loc = base_dir + '/SQL'

# Copy every .csv file in the res_loc dir to SQL recursively

def copy_csv_files(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.csv'):
                src_file = os.path.join(root, file)  # Source file path
                relative_path = os.path.relpath(root, src_dir)  # Get relative path
                dest_folder = os.path.join(dest_dir, relative_path)  # Destination folder
                os.makedirs(dest_folder, exist_ok=True)  # Create destination folder if it doesn't exist
                dest_file = os.path.join(dest_folder, file)  # Destination file path

                shutil.copy2(src_file, dest_file)  # Copy file, preserving metadata
                print(f"Copied: {src_file} -> {dest_file}")

copy_csv_files(res_loc, sql_loc)
