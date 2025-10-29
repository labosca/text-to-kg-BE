import pandas as pd
import json
import pathlib
from difflib import SequenceMatcher
from tqdm import tqdm
import ast

def apply_replacements(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def triple_similarity(triple1, triple2):
    similarities = {}
    for component in ['subject', 'predicate', 'object']:
        value1 = str(triple1[component])
        value2 = str(triple2[component])
        similarities[component] = SequenceMatcher(None, value1, value2).ratio()
    return similarities

def compare_triples(df, col1, col2):
    replacements = {
        'https://w3id.org/rec/full/': 'rec:', 'http://www.example.org/resource/': 'exr:', 
        'http://www.example.org/ontology/': 'exo:', 'http://www.w3.org/2002/07/owl#': 'owl:', 
        'http://www.w3.org/XML/1998/namespace': 'xml:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:', 
        'http://www.w3.org/2001/XMLSchema#': 'xsd:', 'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
    }
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        triple1 = ast.literal_eval(row[col1])
        triple2 = ast.literal_eval(row[col2])
        
        # Apply replacements
        for component in ['subject', 'predicate', 'object']:
            if isinstance(triple1[component], str):
                triple1[component] = apply_replacements(triple1[component], replacements)
            if isinstance(triple2[component], str):
                triple2[component] = apply_replacements(triple2[component], replacements)
        
        similarities = triple_similarity(triple1, triple2)
        overall_similarity = sum(similarities.values()) / 3
        results.append({
            'triple1': triple1,
            'triple2': triple2,
            'subject_similarity': similarities['subject'],
            'predicate_similarity': similarities['predicate'],
            'object_similarity': similarities['object'],
            'overall_similarity': overall_similarity
        })
    
    return pd.DataFrame(results)

current_dir = pathlib.Path(__file__).parent

# Load the Excel file (located in the same folder as this script)
excel_file = current_dir / 'triple_comparison_results.xlsx'
df = pd.read_excel(str(excel_file))

# Identify the correct column names
col1 = 'df2_row'
col2 = 'df1_match'

print(f"\nUsing columns: '{col1}' and '{col2}'")

# Perform the comparison
result = compare_triples(df, col1, col2)

# Analyze the results
threshold = 0.8
close_matches = result[result['overall_similarity'] > threshold]
print("\nClose matches:")
print(close_matches[['triple1', 'triple2', 'subject_similarity', 'predicate_similarity', 'object_similarity', 'overall_similarity']])

# Save results to excel (in the same folder as this script)
output_excel_file = current_dir / 'triple_comparison_results_analyzed.xlsx'
result.to_excel(str(output_excel_file), index=False, engine='openpyxl')
print(f"\nResults saved to '{output_excel_file}'")

# Additional analysis: Average similarities for each component
print("\nAverage similarities:")
print(f"Subject: {result['subject_similarity'].mean():.4f}")
print(f"Predicate: {result['predicate_similarity'].mean():.4f}")
print(f"Object: {result['object_similarity'].mean():.4f}")
print(f"Overall: {result['overall_similarity'].mean():.4f}")

# Distribution of similarities
print("\nDistribution of similarities:")
for component in ['subject', 'predicate', 'object', 'overall']:
    print(f"\n{component.capitalize()} similarity distribution:")
    print(result[f'{component}_similarity'].describe())
