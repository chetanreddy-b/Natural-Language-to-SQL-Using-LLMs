import pandas as pd
import os
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

def extract_query_components(query):
    # Parse the SQL query
    parsed = sqlparse.parse(query)[0]

    # Initialize components
    tables = []
    columns = []
    conditions = []

    # Extract tokens
    for token in parsed.tokens:
        if token.ttype is DML and token.value.upper() == 'SELECT':
            pass  # Handle SELECT clause
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                columns.append(identifier.get_real_name())
        elif isinstance(token, Identifier):
            tables.append(token.get_real_name())
        elif token.ttype is Keyword and token.value.upper() in ('WHERE', 'ORDER BY', 'GROUP BY'):
            conditions.append(token)

    return {
        'tables': set(tables),
        'columns': set(columns),
        'conditions': set(conditions)
    }

def sql_semantic_match(query1, query2):
    # Normalize and extract components from both queries
    components1 = extract_query_components(query1)
    components2 = extract_query_components(query2)

    # Compare tables, columns, and conditions
    table_match = components1['tables'] == components2['tables']
    column_match = components1['columns'] == components2['columns']
    condition_match = components1['conditions'] == components2['conditions']

    # If all components match, the queries are semantically similar
    return table_match and column_match and condition_match

# Example usage
query1 = "SELECT name FROM employees WHERE age > 30 ORDER BY name"
query2 = "SELECT name FROM employees ORDER BY name WHERE age > 30"

print("Queries are semantically equivalent:", sql_semantic_match(query1, query2))

# Paths to each file

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

models = ["phi", "lfm", "gemma_1", "gemma_2", "llama", "mistral"]

# Levenshtein similarity using difflib
def levenshtein_similarity(a, b):
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

def tokenize_sql(query):
    return sqlparse.format(query, keyword_case='upper').split()

# Token-based similarity: Jaccard and Cosine
def jaccard_similarity(query1, query2):
    tokens1 = set(query1.split())
    tokens2 = set(query2.split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

# Cosine_similarity
def cosine_similarity(query1, query2):
    tokens1 = Counter(query1.split())
    tokens2 = Counter(query2.split())
    all_tokens = set(tokens1.keys()).union(set(tokens2.keys()))
    dot_product = sum(tokens1[token] * tokens2[token] for token in all_tokens)
    norm1 = sum(value**2 for value in tokens1.values())**0.5
    norm2 = sum(value**2 for value in tokens2.values())**0.5
    return dot_product / (norm1 * norm2)

# Word Error Rate (WER)
def word_error_rate(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return levenshtein_similarity(reference, hypothesis) / len(ref_tokens)

# Evaluating the df with model
def evaluate_sql(file_path, model):

  df = pd.read_csv(file_path)

  total_queries = len(df)
  results = {
      "levenshtein_similarity": [],
      "jaccard_similarity": [],
      "cosine_similarity": [],
      "word_error_rate": [],
      "exact_match": [],
      "clause_level_precision": [],
      "clause_level_recall": [],
      "clause_level_f1": [],
      "bleu_score": [],
      "rouge_1": [],
      "rouge_2": [],
      "rouge_l": []
  }

  # Initialize ROUGE scorer
  rouge_scorer_tool = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

  for _, row in df.iterrows():
    reference = row["SQL Query"]
    hypothesis = row[model]

    reference_normalized = sqlparse.format(reference, reindent=True, keyword_case='upper')
    hypothesis_normalized = sqlparse.format(hypothesis, reindent=True, keyword_case='upper')

    # Levenshtein Similarity (1 - normalized distance)
    results["levenshtein_similarity"].append(levenshtein_similarity(reference, hypothesis))

    # Jaccard Similarity
    jaccard_sim = jaccard_similarity(reference, hypothesis)
    results["jaccard_similarity"].append(jaccard_sim)

    # Cosine Similarity
    cosine_sim = cosine_similarity(reference, hypothesis)
    results["cosine_similarity"].append(cosine_sim)

    # Word Error Rate
    wer = word_error_rate(reference, hypothesis)
    results["word_error_rate"].append(wer)

    # Exact Match
    exact_match = 1 if reference.strip() == hypothesis.strip() else 0
    results["exact_match"].append(exact_match)

    # Clause-level Comparison (Token-based)
    exact_tokens = set(reference_normalized.split())
    generated_tokens = set(hypothesis_normalized.split())

    # Calculate Precision, Recall, F1
    intersection = exact_tokens & generated_tokens
    precision = len(intersection) / len(generated_tokens) if generated_tokens else 0
    recall = len(intersection) / len(exact_tokens) if exact_tokens else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    results["clause_level_precision"].append(precision)
    results["clause_level_recall"].append(recall)
    results["clause_level_f1"].append(f1_score)

    # Tokenize queries
    reference_tokens = tokenize_sql(reference)
    hypothesis_tokens = tokenize_sql(hypothesis)

    # BLEU Score
    results["bleu_score"].append(sentence_bleu([reference_tokens], hypothesis_tokens))

    # ROUGE Scores
    rouge_result = rouge_scorer_tool.score(reference, hypothesis)
    results["rouge_1"].append(rouge_result["rouge1"].fmeasure)
    results["rouge_2"].append(rouge_result["rouge2"].fmeasure)
    results["rouge_l"].append(rouge_result["rougeL"].fmeasure)

  # Return average metrics
  Metrics = {}
  for metric, values in results.items():
    Metrics[metric] = round(sum(values) / total_queries, 3)
  return Metrics

metric_path = base_dir + '/Metrics/metrics.json'

Metric_Results = {}

for key, value in sql_path.items():
  print("Working on " + key)
  file_results = {}
  for model in models:
    path = sql_dir + value
    file_results[model] = evaluate_sql(path, model)
    print(model, end = " ")
  Metric_Results[key] = file_results
  print("Done")
  
# Save to a JSON file
with open(metric_path, "w") as file:
    json.dump(Metric_Results, file)
    
datasets = ["Crime_Simple", "Crime_Complex", "Shooting_Simple", "Shooting_Complex", "Housing_Simple", "Housing_Complex", "Simple", "Complete", "Complex"]

metric_df_dir = base_dir + '/Metrics'

for dataset in datasets:
  print("Working on " + dataset)
  df = pd.DataFrame(Metric_Results[dataset]).transpose()
  df.to_csv(metric_df_dir + '/' + dataset + '_metrics.csv')
  print("Done")