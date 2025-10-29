import json, os
import pandas as pd
import pathlib
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm

#Format the json files
def flatten_json(json_data):
    flattened_data = []
    for item in json_data:
        subject = item.get('@id', '')
        for key, value in item.items():
            if key not in ['@id', '@context']:
                if isinstance(value, list):
                    for v in value:
                        flattened_data.append({'subject': subject, 'predicate': key, 'object': v})
                else:
                    flattened_data.append({'subject': subject, 'predicate': key, 'object': value})
    return pd.DataFrame(flattened_data)

#Create a similarity function to compare triples
def triple_similarity(row1, row2):
    similarity = 0
    for col in ['subject', 'predicate', 'object']:
        similarity += SequenceMatcher(None, str(row1[col]), str(row2[col])).ratio()
    return similarity / 3


#Implement the comparison logic
def find_closest_matches(df1, df2):
    replacements = {
        'https://w3id.org/rec/full/': 'rec:', 'http://www.example.org/resource/': 'exr:', 
        'http://www.example.org/ontology/': 'exo:', 'http://www.w3.org/2002/07/owl#': 'owl:', 
        'http://www.w3.org/XML/1998/namespace': 'xml:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:', 
        'http://www.w3.org/2001/XMLSchema#': 'xsd:', 'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
    }
    
    df1 = df1.replace(replacements, regex=True)
    df2 = df2.replace(replacements, regex=True)
    
    matches = []
    for _, row2 in tqdm(df2.iterrows(), total=len(df2)):
        best_match = None
        best_similarity = 0
        for _, row1 in df1.iterrows():
            similarity = triple_similarity(row2, row1)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = row1
        matches.append({
            'df2_row': row2.to_dict(),
            'df1_match': best_match.to_dict() if best_match is not None else None,
            'similarity': best_similarity
        })
    
    result_df = pd.DataFrame(matches)
    return result_df

# Load and flatten the JSON files
current_dir = pathlib.Path(__file__).parent

with open(str(current_dir / 'ground1.json'), 'r') as f:
    df1 = flatten_json(json.load(f))

with open(str(current_dir / 'false_positive_full_text.json'), 'r') as f:
    df2 = flatten_json(json.load(f))


# Print DataFrame information for debugging
print("DataFrame 1 columns:", df1.columns)
print("DataFrame 2 columns:", df2.columns)
print("\nDataFrame 1 first few rows:")
print(df1.head())
print("\nDataFrame 2 first few rows:")
print(df2.head())

# Perform the comparison
result = find_closest_matches(df1, df2)

#Analyze the results
# Display matches with similarity above a threshold
threshold = 0.8
close_matches = result[result['similarity'] > threshold]
print("\nClose matches:")
print(close_matches)

# Save results to excel
excel_file = current_dir / 'triple_comparison_results.xlsx'
result.to_excel(str(excel_file), index=False, engine='openpyxl')
print(f"\nResults saved to '{excel_file}'")

######################################################
######################################################

# Calculate similarity between entities of triples
import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from difflib import SequenceMatcher

# Function to calculate string similarity
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to compare triples
def compare_triples(row):
    triple1 = row.iloc[0]
    triple2 = row.iloc[1]
    
    subject_similarity = calculate_similarity(triple1['subject'], triple2['subject'])
    predicate_similarity = calculate_similarity(triple1['predicate'], triple2['predicate'])
    object_similarity = calculate_similarity(triple1['object'], triple2['object'])
    
    return pd.Series({
        'subject_similarity': subject_similarity,
        'predicate_similarity': predicate_similarity,
        'object_similarity': object_similarity
    })

# Load the Excel file
df = pd.read_excel(str(current_dir / 'triple_comparison_results.xlsx'))

# Evaluate similarity between triples in the first two columns
df[['subject_similarity', 'predicate_similarity', 'object_similarity']] = df.apply(compare_triples, axis=1)

# Find rows where all parts of the triple are similar (e.g., similarity > 0.8)
similar_triples = df[(df['subject_similarity'] > 0.8) & 
                     (df['predicate_similarity'] > 0.8) & 
                     (df['object_similarity'] > 0.8)]

print("Triples with high similarity:")
print(similar_triples)

# Continue with the rest of your script...
# (e.g., querying the graph, saving results, etc.)