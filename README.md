# Natural Language to SQL queries Using LLMs

This repository evaluates the performance of various Large Language Models (LLMs) for generating SQL queries based on natural language inputs. By implementing prompt engineering techniques (Zero-Shot, One-Shot, Few-Shot), the project analyzes the usability of LLMs for social science research involving the NORP database.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
   - [Requirements](#requirements)
   - [Installation](#installation)
4. [Directory Structure](#directory-structure)
5. [Methodology](#methodology)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Prompt Engineering Approaches](#prompt-engineering-approaches)
6. [Results and Observations](#results-and-observations)
7. [References](#references)



## Introduction

The NORP LLM Chatbot bridges the gap between non-technical users and database management by translating natural language queries into SQL commands. Social scientists, lacking SQL expertise, can seamlessly query the NORP database for actionable insights.


## Features

- **Natural Language to SQL**: Converts user-friendly queries into executable SQL commands.
- **Multi-Model Evaluation**: Supports LLMs like Llama, Gemma, Phi-3, Mistral, and LFM.
- **Prompt Engineering Techniques**: Explores and benchmarks Zero-shot, One-shot, and Few-shot approaches.
- **Performance Metrics**: Evaluates SQL generation accuracy using BLEU, ROUGE, Levenshtein, and execution feedback.
- **Metabase Integration**: Validates generated queries by running them in Metabase.


## Setup and Installation

### Requirements

- Python 3.8+
- Libraries:
  - `openai`
  - `sqlparse`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `nltk`
  - `rouge-score`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chetanreddy-b/Natural-Language-to-SQL-Using-LLMs
   cd norp-llm-chatbot-evaluation
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
Set up API keys for LLMs in the environment or directly in the code.

### Directory Structure

```plaintext
NORP-LLM-Evaluation/
├── Code/                    # Python scripts for model integration and evaluation
├── Datasets/                # Sample datasets used for query generation
├── Results/                 # Metrics and evaluation results
├── Images/                  # Visualizations and screenshots
├── Prompt_Engineering/      # Prompt experiments and examples
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```
## Methodology

The evaluation process for LLMs involved converting natural language queries into SQL commands and validating their execution against the NORP Metabase. The following metrics were used to evaluate the models:

### Evaluation Metrics

| **Metric**               | **Description**                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------|
| **BLEU**                 | Measures the precision of sequences in the generated queries compared to the ideal SQL query.     |
| **ROUGE**                | Assesses recall by comparing the overlap between generated queries and the reference query.       |
| **Levenshtein Similarity** | Evaluates edit distance, quantifying how many insertions, deletions, or substitutions are needed. |
| **Jaccard Similarity**   | Measures intersection over union of unique tokens between the generated and ideal queries.        |
| **Cosine Similarity**    | Computes the similarity between vector representations of queries based on their semantic meaning.|

### Prompt Engineering Approaches

| **Approach**    | **Details**                                                                                 | **Performance**                            |
|------------------|---------------------------------------------------------------------------------------------|--------------------------------------------|
| **Zero-Shot**   | Provides minimal context; only the natural language query is given.                         | Quick but inaccurate SQL generation.       |
| **One-Shot**    | Includes schema details to improve context.                                                 | Better structure but struggles with nuances.|
| **Few-Shot**    | Adds schema details and sample rows for richer context.                                     | Best results with high accuracy.           |

### Results and Observations

1. **Prompt Engineering**: Few-shot prompting outperformed zero and one-shot approaches, generating queries that closely matched the ideal SQL.
2. **Performance vs. Latency**:
   - **LFM MoE** exhibited high performance but at the cost of significant latency.
   - **Gemma1** struck a balance between performance and response time.
3. **Complex Queries**:
   - Highlighted limitations of LLMs in handling operations like joins, CASE statements, and multi-column logic.
   - Semantic understanding of data relationships remains a challenge for general-purpose LLMs.


## References
- Qingxiu Dong et al., A Survey on In-context Learning
- Pranab Sahoo et al., Prompt Engineering in Large Language Models
- Qinggang Zhang1 et al., Structure Guided Large Language Model for SQL Generation

