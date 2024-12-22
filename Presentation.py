
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base_dir = '/content/drive/MyDrive/RTS_Chetan_Chathurvedhi'

df = pd.read_csv(base_dir + '/SQL/latency_comp.csv')

simple_count = 88
complex_count = 40

# gemma_simple_lat = 182.85
# gemma_complex_lat = 110.90

# lfm_simple_lat = 443.54
# lfm_complex_lat = 252.81

# gemma_simple_bleu = 0.016
# lfm_simple_bleu = 0.026

# gemma_complex_bleu = 0.006
# lfm_complex_bleu = 0.007

# lfm_complete_lat = lfm_complex_lat + lfm_simple_lat
# gemma_complete_lat = gemma_complex_lat + gemma_simple_lat

# lfm_complete_bleu = 0.02
# gemma_complete_bleu = 0.012

# Data
categories = ['Simple', 'Complex', 'Complete']
latencies_lfm = [443.54/88, 252.81/40, (443.54 + 252.81)/128]
latencies_gemma = [182.85/88, 110.90/40, (182.85 + 110.90)/128]
bleu_scores_lfm = [0.026, 0.007, 0.02]
bleu_scores_gemma = [0.016, 0.006, 0.012]

# Scaling the latencies by count


x = np.arange(len(categories))  # label locations
width = 0.35  # width of the bars

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Latency bars
bar1 = ax1.bar(x - width/2, latencies_lfm, width, label='LFM Latency', color='skyblue')
bar2 = ax1.bar(x + width/2, latencies_gemma, width, label='GEMMA Latency', color='salmon')

# Set up the primary y-axis for latency
ax1.set_xlabel('Categories')
ax1.set_ylabel('Latency (ms)', color='black')
ax1.set_title('Latency vs Performance for LFM and GEMMA')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend(loc='upper left')

# BLEU scores
ax2 = ax1.twinx()  # Secondary y-axis for BLEU scores
ax2.plot(x, bleu_scores_lfm, marker='o', label='LFM BLEU', color='blue')
ax2.plot(x, bleu_scores_gemma, marker='s', label='GEMMA BLEU', color='red')
ax2.set_ylabel('BLEU Score', color='black')

# Combine legends from both axes
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Display the plot
plt.tight_layout()
plt.savefig(base_dir + '/Images/latency_bleu_comp.png')

# Scale BLEU scores to centi-BLEU
bleu_scores_lfm_scaled = [score * 100 for score in bleu_scores_lfm]
bleu_scores_gemma_scaled = [score * 100 for score in bleu_scores_gemma]

# Scatter plot
plt.figure(figsize=(8, 6))

# Scatter points
plt.scatter(latencies_lfm, bleu_scores_lfm_scaled, color='blue', label='LFM')
plt.scatter(latencies_gemma, bleu_scores_gemma_scaled, color='red', label='GEMMA')

# Add lines connecting points of the same category
for i, category in enumerate(categories):
    plt.plot(
        [latencies_lfm[i], latencies_gemma[i]],
        [bleu_scores_lfm_scaled[i], bleu_scores_gemma_scaled[i]],
        color='gray', linestyle='--', linewidth=1
    )
    plt.text((latencies_lfm[i] + latencies_gemma[i]) / 2,
             (bleu_scores_lfm_scaled[i] + bleu_scores_gemma_scaled[i]) / 2,
             category, fontsize=9, ha='center', va='center')

# Labels and title
plt.xlabel('Latency (ms)')
plt.ylabel('BLEU Score (x 100)')
plt.title('Latency vs BLEU Score (Scaled) for LFM and GEMMA')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig(base_dir + '/Images/latency_bleu_scatter.png')

import json

metrics_loc = base_dir + '/Metrics/metrics.json'
Metric_Results = {}

with open(metrics_loc, "r") as file:
  Metric_Results = json.load(file)

Simple_Metrics = Metric_Results["Simple"]
Complex_Metrics = Metric_Results["Complex"]
Complete_Metrics = Metric_Results["Complete"]

simple_df = pd.DataFrame(Simple_Metrics).transpose()
complex_df = pd.DataFrame(Complex_Metrics).transpose()
complete_df = pd.DataFrame(Complete_Metrics).transpose()

