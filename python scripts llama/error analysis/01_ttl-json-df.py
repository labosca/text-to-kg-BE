import json, os, glob, openpyxl
import pathlib
import pandas as pd
from rdflib import Graph, Namespace

# Specify the input folder containing TTL files
# From inside this `error analysis` folder, go up one level and use the `results` subfolder
current_dir = pathlib.Path(__file__).parent
input_folder = current_dir.parent / "results"

# Specify the output folder (keep current folder where this script resides)
output_folder = current_dir

# Define the pattern for TTL files with certain names
pattern = os.path.join(str(input_folder), "*false_positive_*.ttl")

# Ensure the output folder exists
os.makedirs(str(output_folder), exist_ok=True)

# Function to process a single TTL file
def process_ttl_file(file_path):
    # Load the TTL file
    g = Graph()
    g.parse(file_path, format="turtle")

    # Convert to JSON-LD
    json_ld = g.serialize(format="json-ld", indent=2)

    # Parse the JSON-LD string to a Python dictionary
    json_data = json.loads(json_ld)

    # Create the full path for the output JSON file
    json_output_file = os.path.join(str(output_folder), f"{os.path.splitext(os.path.basename(file_path))[0]}.json")

    # Save the JSON data to the specified file
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON file saved as '{json_output_file}'")

    # Flatten the JSON structure
    flattened_data = []
    for item in json_data:
        subject = item.get('@id', '')
        for key, value in item.items():
            if key != '@id' and key != '@context':
                if isinstance(value, list):
                    for v in value:
                        flattened_data.append({'subject': subject, 'predicate': key, 'object': v})
                else:
                    flattened_data.append({'subject': subject, 'predicate': key, 'object': value})

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    replacements = {
        'https://w3id.org/rec/full/': 'rec:', 'http://www.example.org/resource/': 'exr:', 
        'http://www.example.org/ontology/': 'exo:', 'http://www.w3.org/2002/07/owl#': 'owl:', 
        'http://www.w3.org/XML/1998/namespace': 'xml:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:', 
        'http://www.w3.org/2001/XMLSchema#': 'xsd:', 'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
    }
    df = df.replace(replacements, regex=True)
    return df

# Dictionary to store DataFrames
dataframes = {}

# Iterate through the matching files
for i, file_path in enumerate(glob.glob(pattern), start=1):
    print(f"Processing: {file_path}")
    df = process_ttl_file(file_path)
    
    # Name the DataFrame with an incremental number
    df_name = f"df{i}"
    dataframes[df_name] = df
    
    # Save DataFrame to Excel
    # Create a Pandas Excel writer using XlsxWriter as the engine
    excel_output_file = os.path.join(str(output_folder), "fp-all_df.xlsx")
    with pd.ExcelWriter(excel_output_file, engine="xlsxwriter") as writer:
        # Iterate through the dataframes dictionary
        for df_name, df in dataframes.items():
            # Write each DataFrame to a different sheet
            df.to_excel(writer, sheet_name=df_name, index=False)

    writer.close()

    print(f"All DataFrames saved to '{excel_output_file}'")

# Now you can access the DataFrames using their names, e.g., dataframes['df1'], dataframes['df2'], etc.
