# %%
import subprocess
import sys

def install_required_packages():
    packages = [
        'bitsandbytes',
        'transformers',
        'torch',
        'rdflib'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment to install required packages
# install_required_packages()

from huggingface_hub import login
import os

# Load token from environment variable or config file
hf_token = os.getenv('HUGGINGFACE_TOKEN')  # Set this environment variable with your token
if hf_token:
    login(hf_token)

# %%

from glob import glob
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import transformers
import torch
import shutil
import json

from transformers import BitsAndBytesConfig
from glob import glob
from tqdm import tqdm
import re

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


# %%
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Check if CUDA is available before using it
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA not available, using CPU instead")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# %%
# Read SYSTEM2 content from example.txt
example_path = pathlib.Path(__file__).parent.parent / "python scripts llama" / "input" / "example.txt"
with open(example_path, 'r', encoding='utf-8') as f:
    SYSTEM2 = f.read()

# %%
# Read INPUT content from input.txt
input_path = pathlib.Path(__file__).parent.parent / "python scripts llama" / "input" / "input.txt"
with open(input_path, 'r', encoding='utf-8') as f:
    INPUT = f.read()

# %%
import time

# %%
#infer Meta-Llama-3.1-8B-Instruct

messages = [
    {"role": "system", "content": SYSTEM2},
    {"role": "user", "content": INPUT}
     ]

start = time.time()
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=128256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.4,
)
response = outputs[0][input_ids.shape[-1]:]
model_output = tokenizer.decode(response, skip_special_tokens=True)
exec_time = time.time() - start
print(model_output)

# %%
num_tokens = len(tokenizer.decode(response))

print(f"Number of tokens in the output: {num_tokens}")

print(f"Execution time: {exec_time} seconds")

# %%
from rdflib import Graph

import pathlib

# Use pathlib for cross-platform path handling
current_dir = pathlib.Path(__file__).parent
ground_path = current_dir / "input" / "ground1.ttl"
ground_graph = Graph()
ground_graph.parse(str(ground_path))
print(ground_graph)

# %%
import re 

#start_ticks = re.search("```turtle", model_output)
#end_ticks = re.search("```", model_output[start_ticks.span()[1]:])
#rdf_output = model_output[start_ticks.span()[1] : start_ticks.span()[1] + end_ticks.span()[0]]
rdf_output = model_output

ttl_path = current_dir / "results" / "rdf_output_fulltext.ttl"
try:
    if ttl_path.exists():
        ttl_path.unlink()
    ttl_path.write_text(rdf_output)
    output_graph = Graph()
    output_graph.parse(str(ttl_path))
    print("Output graph loaded successfully with", len(output_graph), "triples")
except Exception as e:
    print(f"Error processing RDF output: {e}")
    raise



# %%
print("len ground_graph", len(ground_graph))
print("len output_graph", len(output_graph))

true_positive = ground_graph & output_graph
true_positive.serialize(destination=str(current_dir / "results" / "true_positive_full_text.ttl"), format="ttl")
print("True positives:", len(true_positive))

# %%
false_positive = output_graph - true_positive
false_positive.serialize(destination=str(current_dir / "results" / "false_positive_full_text.ttl"), format="ttl")
print("False positives:", len(false_positive))

# %%
false_negative = ground_graph - true_positive
false_negative.serialize(destination=str(current_dir / "results" / "false_negative_full_text.ttl"), format="ttl")
print(len(false_negative))

# %%
precision = len(true_positive) / len(output_graph)
recall = len(true_positive) / len(ground_graph)
f1 = 2 * ( precision * recall ) / ( precision + recall ) if precision + recall != 0 else 0

print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

# %%
