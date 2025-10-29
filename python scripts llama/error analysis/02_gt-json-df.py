import json, os
import pathlib
import pandas as pd
from rdflib import Graph

# Input and output paths
# Point input_file to the sibling `input/ground1.ttl` relative to this script
current_dir = pathlib.Path(__file__).parent
input_file = current_dir.parent / "input" / "ground1.ttl"
output_folder = current_dir
os.makedirs(str(output_folder), exist_ok=True)

# Remove @prefix lines from input file
with open(str(input_file), 'r') as file:
    lines = file.readlines()

cleaned_input = os.path.join(str(output_folder), "cleaned_input.ttl")
with open(cleaned_input, 'w') as file:
    for line in lines:
        if not line.strip().startswith("@prefix"):
            file.write(line)

# Process TTL file
g = Graph()
g.parse(str(input_file), format="turtle")
json_data = json.loads(g.serialize(format="json-ld", indent=2))

# Flatten JSON structure
flattened_data = []
for item in json_data:
    subject = item.get('@id', '')
    for key, value in item.items():
        if key not in ['@id', '@context']:
            if isinstance(value, list):
                flattened_data.extend({'subject': subject, 'predicate': key, 'object': v} for v in value)
            else:
                flattened_data.append({'subject': subject, 'predicate': key, 'object': value})

# Create and process DataFrame
df = pd.DataFrame(flattened_data)
replacements = {
    'https://w3id.org/rec/full/': 'rec:', 'http://www.example.org/resource/': 'exr:', 
    'http://www.example.org/ontology/': 'exo:', 'http://www.w3.org/2002/07/owl#': 'owl:', 
    'http://www.w3.org/XML/1998/namespace': 'xml:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:', 
    'http://www.w3.org/2001/XMLSchema#': 'xsd:', 'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
}
df = df.replace(replacements, regex=True)

# Save outputs
base_name = os.path.splitext(os.path.basename(str(input_file)))[0]
json_output = os.path.join(str(output_folder), f"{base_name}.json")
excel_output = os.path.join(str(output_folder), f"{base_name}.xlsx")

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

df.to_excel(excel_output, index=False)

print(f"JSON saved as '{json_output}'")
print(f"Excel saved as '{excel_output}'")
