import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

##########################################
#CONFUSION MATRIX LLM
##########################################

# Data for the confusion matrix
scenarios = ["Full-text", "Paragraphs", "Sentences"] #edit the values accordingly to the results from each scenario (fulltext, paragraphs, sentences)
tp = [80, 71, 65]
fp = [32, 320, 757]
fn = [107, 116, 122]

# Constructing the confusion matrix
conf_matrix = np.array([tp, fp, fn])

# Creating a DataFrame for visualization
df_cm = pd.DataFrame(conf_matrix, index=["True Positives", "False Positives", "False Negatives"], columns=scenarios)

# Plotting the confusion matrix
current_dir = pathlib.Path(__file__).parent
tables_dir = current_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, annot_kws={"fontsize": 8})
plt.title("Confusion Matrix by Inference Scenarios", fontsize=12, pad=20)
plt.xlabel("Inference Scenario", fontsize=10, fontweight="bold")
plt.ylabel("Metric", fontsize=10, fontweight="bold")
plt.savefig(str(tables_dir / 'confusion_matrix.jpg'), dpi=600, bbox_inches="tight")  # Save as PNG
plt.show()

'''For each scenario:

    Normalized Value = Metric Value / Total Extracted Triples

where:
    Full text: 112 extracted triples
    Paragraphs: 391 extracted triples
    Sentences: 822 extracted triples'''

# Total extracted triples for each scenario
total_extracted = np.array([112, 391, 822])

# Normalizing the confusion matrix
normalized_conf_matrix = conf_matrix / total_extracted

# Creating a DataFrame for visualization
df_cm_normalized = pd.DataFrame(
    normalized_conf_matrix, 
    index=["True Positives", "False Positives", "False Negatives"], 
    columns=scenarios
)

# Plotting the normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, annot_kws={"fontsize": 8})
plt.title("Normalized Confusion Matrix (Per Extracted Triples)", fontsize=12, pad=20)
plt.xlabel("Inference Scenario", fontsize=10, fontweight="bold")
plt.ylabel("Metric", fontsize=10, fontweight="bold")
plt.savefig(str(tables_dir / 'confusion_matrix_normalized.jpg'), dpi=600, bbox_inches="tight")  # Save as PNG
plt.show()

# Save to an Excel file with two sheets
with pd.ExcelWriter(str(tables_dir / 'confusion_matrices.xlsx')) as writer:
    df_cm.to_excel(writer, sheet_name="Original Matrix")
    df_cm_normalized.to_excel(writer, sheet_name="Normalized Matrix")

print("Confusion matrices saved as 'confusion_matrices.xlsx'")


##########################################
#CONFUSION MATRIX FPs
##########################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Excel file (update the filename and sheet name if needed)
file_path = current_dir / 'triple_comparison_results_analyzed.xlsx'
df = pd.read_excel(str(file_path), sheet_name="Sheet1")  

# Convert numeric columns (handling commas)
for col in ["subject_similarity", "predicate_similarity", "object_similarity", "overall_similarity"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Rename columns for shorter labels
df = df.rename(columns={
    "subject_similarity": "Subject S.",
    "predicate_similarity": "Predicate S.",
    "object_similarity": "Object S.",
    "overall_similarity": "Overall S."
})

# Select relevant columns
df_similarity = df[["Subject S.", "Predicate S.", "Object S.", "Overall S."]]

# Set index to triple pairs for better visualization
df_similarity.index = [f"FP_{i+1}" for i in range(len(df))]

# Create Heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(df_similarity, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, annot_kws={"fontsize": 8})

# Adjust label size and orientation
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=0)  # Horizontal x-axis labels
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)  # Smaller y-axis labels

plt.title("FP Triples Similarity Heatmap", fontsize=12, pad=20)
plt.xlabel("Similarity Metric", fontsize=10, fontweight="bold")
plt.ylabel("FP Triples", fontsize=10, fontweight="bold")
plt.savefig(str(tables_dir / 'confusion_matrix_fp.jpg'), dpi=600, bbox_inches="tight")  # Save as PNG
plt.show()


# Sort DataFrame by "Overall S." in descending order
df_sorted = df.sort_values(by="Overall S.", ascending=False)

# Select relevant columns
df_similarity = df_sorted[["Subject S.", "Predicate S.", "Object S.", "Overall S."]]

# Set index as Triples (after sorting)
df_similarity.index = [f"FP_{i+1}" for i in range(len(df))]

# Create Heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(df_similarity, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, annot_kws={"fontsize": 8})

# Adjust label size and orientation
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=0)  # Horizontal x-axis labels
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)  # Smaller y-axis labels

plt.title("FP Triples Similarity Heatmap (Sorted by Overall Similarity)", fontsize=12, pad=20)
plt.xlabel("Similarity Metric", fontsize=10, fontweight="bold")
plt.ylabel("FP Triples", fontsize=10, fontweight="bold")
plt.savefig(str(tables_dir / 'confusion_matrix_fp_sorted.jpg'), dpi=600, bbox_inches="tight")  # Save as PNG
plt.show()

'''
# Create a bar chart for similarity comparison
df_similarity.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")

# Customize labels and title
plt.title("Similarity Comparison Across Triple Pairs", fontsize=12)
plt.ylabel("Similarity Score", fontsize=10)
plt.xlabel("Triple Pair", fontsize=10)
plt.xticks(rotation=45, fontsize=8)  # Rotate x-axis labels slightly
plt.legend(title="Similarity Metric", fontsize=8)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
'''