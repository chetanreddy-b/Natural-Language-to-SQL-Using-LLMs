import pandas as pd
import openai
import os
from groq import Groq
import chardet
import shutil
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import difflib
# import ntlk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.util import ngrams
from collections import Counter
import json
import warnings
warnings.filterwarnings("ignore")

from SQL_Gen_Groq import generate_sql_query

base_dir = '/content/drive/MyDrive/RTS_Chetan_Chathurvedhi'

sql_dir = base_dir + '/SQL'

sql_path = {
    "Crime_Simple":'/crime_simple.csv',
    "Crime_Complex":'/crime_complex.csv',
    "Shooting_Simple":'/shooting_simple.csv',
    "Shooting_Complex":'/shooting_complex.csv',
    "Housing_Simple":'/housing_simple.csv',
    "Housing_Complex":'/housing_complex.csv',
    "Complete":'/complete.csv',
    "Simple":'/simple.csv',
    "Complex":'/complex.csv'
}

df_crime_simple = pd.read_csv(sql_dir+sql_path["Crime_Simple"])
df_crime_complex = pd.read_csv(sql_dir+sql_path["Crime_Complex"])
df_shooting_simple = pd.read_csv(sql_dir+sql_path["Shooting_Simple"])
df_shooting_complex = pd.read_csv(sql_dir+sql_path["Shooting_Complex"])
df_housing_simple = pd.read_csv(sql_dir+sql_path["Housing_Simple"])
df_housing_complex = pd.read_csv(sql_dir+sql_path["Housing_Complex"])

df_simple = pd.concat([df_crime_simple, df_shooting_simple, df_housing_simple], ignore_index=True)
df_complex = pd.concat([df_crime_complex, df_shooting_complex, df_housing_complex], ignore_index=True)

df_complete = pd.concat([df_simple, df_complex], ignore_index=True)

# Save the dataframes

df_simple.to_csv(sql_dir+'/simple.csv', index=False)
df_complex.to_csv(sql_dir+'/complex.csv', index=False)
df_complete.to_csv(sql_dir+'/complete.csv', index=False)

import time

start_time = time.perf_counter()

df_temp = df_complete.copy()

df_temp["temp"] = df_temp.apply(lambda row: generate_sql_query("gemma_1", row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)

end_time = time.perf_counter()

print(f"Time taken for Gemma 1: {end_time - start_time:0.4f} seconds")

start_time = time.perf_counter()

df_temp['temp'] = df_temp.apply(lambda row: generate_sql_query("gemma_2", row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)

end_time = time.perf_counter()

print(f"Time taken for Gemma 2: {end_time - start_time:0.4f} seconds")

start_time = time.perf_counter()

df_temp['temp'] = df_temp.apply(lambda row: generate_sql_query("lfm", row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2), axis=1)

end_time = time.perf_counter()

print(f"Time taken for LFM: {end_time - start_time:0.4f} seconds")

# Gemma

sql_loc = base_dir + '/SQL'

df_simple = pd.read_csv(sql_loc+'/simple.csv',encoding='utf-8')
df_complex = pd.read_csv(sql_loc+'/complex.csv',encoding='windows-1254')

df_simple['complexity'] = -1
df_complex['complexity'] = 1

df_complete = pd.concat([df_simple, df_complex], ignore_index=True)

import numpy as np
import time
from rouge_score import rouge_scorer

df_complete['Gemma_Latency'] = np.nan
df_complete['Gemma_Rouge'] = np.nan
df_complete['Gemma_Levenshtein'] = np.nan
df_complete['gemma_lat'] = np.nan

# Levenshtein similarity using difflib
def levenshtein_similarity(a, b):
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

rouge_scorer_tool = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

for index, row in df_complete.iterrows():
  start_time = time.perf_counter()
  df_complete.at[index, 'gemma_lat'] = generate_sql_query("gemma_1", row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2)
  end_time = time.perf_counter()
  df_complete.at[index, 'Gemma_Latency'] = end_time - start_time
  df_complete.at[index, 'Gemma_Levenshtein'] = levenshtein_similarity(row['SQL Query'], df_complete.at[index, 'gemma_lat'])
  df_complete.at[index, 'Gemma_Rouge'] = rouge_scorer_tool.score(row['SQL Query'], df_complete.at[index, 'gemma_lat'])['rouge1'].fmeasure

df_complete.to_csv(sql_loc+'/latency_comp.csv', index=False)

import matplotlib.pyplot as plt

image_dir = base_dir + '/Images'


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df_complete[df_complete['complexity'] == -1]['Gemma_Levenshtein'], df_complete[df_complete['complexity'] == -1]['Gemma_Latency'], color='blue', label='Simple')
plt.scatter(df_complete[df_complete['complexity'] == 1]['Gemma_Levenshtein'], df_complete[df_complete['complexity'] == 1]['Gemma_Latency'], color='red', label='Complex')

plt.xlabel('Levenshtein Similarity')
plt.ylabel('Latency (seconds)')
plt.title('Levenshtein Similarity vs. Latency')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(image_dir + '\gemma_1_levenshtein')


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df_complete[df_complete['complexity'] == -1]['Gemma_Rouge'], df_complete[df_complete['complexity'] == -1]['Gemma_Latency'], color='blue', label='Simple')
plt.scatter(df_complete[df_complete['complexity'] == 1]['Gemma_Rouge'], df_complete[df_complete['complexity'] == 1]['Gemma_Latency'], color='red', label='Complex')

plt.xlabel('Rouge Similarity')
plt.ylabel('Latency (seconds)')
plt.title('Rouge Similarity vs. Latency')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(image_dir + '\gemma_1_rouge')

# Now for LFM

import numpy as np
import time
from rouge_score import rouge_scorer

df_complete['Latency'] = np.nan
df_complete['Rouge'] = np.nan
df_complete['Levenshtein'] = np.nan

# Levenshtein similarity using difflib
def levenshtein_similarity(a, b):
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

rouge_scorer_tool = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

for index, row in df_complete.iterrows():
  start_time = time.perf_counter()
  df_complete['sql_gen'] = generate_sql_query("lfm", row['Natural Language Query'], row['Schema'], row['Top 5 Entries of Table'], 2)
  end_time = time.perf_counter()
  df_complete.at[index, 'Latency'] = end_time - start_time
  df_complete.at[index, 'Levenshtein'] = levenshtein_similarity(row['SQL Query'], df_complete.at[index, 'sql_gen'])
  df_complete.at[index, 'Rouge'] = rouge_scorer_tool.score(row['SQL Query'], df_complete.at[index, 'sql_gen'])['rouge1'].fmeasure

import matplotlib.pyplot as plt

image_dir = base_dir + '/Images'


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df_complete[df_complete['complexity'] == -1]['Levenshtein'], df_complete[df_complete['complexity'] == -1]['Latency'], color='blue', label='Simple')
plt.scatter(df_complete[df_complete['complexity'] == 1]['Levenshtein'], df_complete[df_complete['complexity'] == 1]['Latency'], color='red', label='Complex')

plt.xlabel('Levenshtein Similarity')
plt.ylabel('Latency (seconds)')
plt.title('Levenshtein Similarity vs. Latency')
plt.legend()
plt.grid(True)
# plt.show()

plt.savefig(image_dir + '\lfm_levenshtein')


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df_complete[df_complete['complexity'] == -1]['Rouge'], df_complete[df_complete['complexity'] == -1]['Latency'], color='blue', label='Simple')
plt.scatter(df_complete[df_complete['complexity'] == 1]['Rouge'], df_complete[df_complete['complexity'] == 1]['Latency'], color='red', label='Complex')

plt.xlabel('Rouge Similarity')
plt.ylabel('Latency (seconds)')
plt.title('Rouge Similarity vs. Latency')
plt.legend()
plt.grid(True)
# plt.show()

plt.savefig(image_dir + '\lfm_rouge')