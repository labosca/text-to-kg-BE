# %%
import subprocess
import sys
import os
import pathlib
import time
import re
from glob import glob
import transformers
import torch
import shutil
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rdflib import Graph
import subprocess
import sys
import os
import pathlib
import time
import re
from glob import glob
import transformers
import torch
import shutil
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rdflib import Graph

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

# %%
# Model configuration
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


# %%
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA not available, using CPU instead")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# %%
# Load input files
current_dir = pathlib.Path(__file__).parent
example_path = current_dir / "input" / "example.txt"
input_path = current_dir / "input" / "input.txt"

# Read system prompt
with open(example_path, 'r', encoding='utf-8') as f:
    SYSTEM2 = f.read()

# Read input text
with open(input_path, 'r', encoding='utf-8') as f:
    INPUT = f.read()

# %%
try:
    from huggingface_hub import login
except ImportError:
    print("Warning: huggingface_hub not installed. Will attempt to install it.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import login

# Load token from environment variable or config file
hf_token = os.getenv('HUGGINGFACE_TOKEN')  # Set this environment variable with your token
if hf_token:
    try:
        login(hf_token)
    except Exception as e:
        print(f"Warning: could not login to HuggingFace Hub: {e}")

import time

# %%
# Process text by sentences
def split_into_sentences(text):
    # Simple sentence splitter — preserves periods
    sentences = [s.strip() + '.' for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences

# Process input and generate output
model_outputs = []
start = time.time()
num_tokens = 0

sentences = split_into_sentences(INPUT)
print(f"Processing {len(sentences)} sentences...")

for i, text in enumerate(sentences, 1):
    print(f"\\nProcessing sentence {i}/{len(sentences)}")
    print("Input:", text[:120] + "..." if len(text) > 120 else text)
    
    messages = [
        {"role": "system", "content": SYSTEM2},
        {"role": "user", "content": text}
    ]
    
    try:
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
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.4,
        )
        response = outputs[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(response, skip_special_tokens=True)
        model_outputs.append(output_text)
        num_tokens += len(tokenizer.decode(response))
        print("Output generated successfully")
    except Exception as e:
        print(f"Error processing sentence {i}: {e}")
        continue

exec_time = time.time() - start
print(f"\\nProcessing complete:")
print(f"Number of tokens in the output: {num_tokens}")
print(f"Execution time: {exec_time} seconds")

# %%
# Process and save results
def process_outputs(model_outputs, current_dir):
    # Load ground truth
    ground_path = current_dir.parent / "input" / "ground1.ttl"
    ground_graph = Graph()
    try:
        ground_graph.parse(str(ground_path))
        print(f"Ground truth graph loaded with {len(ground_graph)} triples")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        raise

    # Combine outputs and save
    rdf_output = "\\n\\n".join(model_outputs)
    ttl_path = current_dir / "results" / "rdf_output_sentence.ttl"
    output_graph = Graph()
    try:
        if ttl_path.exists():
            ttl_path.unlink()
        ttl_path.write_text(rdf_output)
        
        output_graph.parse(str(ttl_path))
        print(f"Output graph loaded successfully with {len(output_graph)} triples")
    except Exception as e:
        print(f"Error processing RDF output: {e}")
        raise
    
    return ground_graph, output_graph

# Process outputs and calculate metrics
ground_graph, output_graph = process_outputs(model_outputs, current_dir)

# Calculate metrics
print("\\nMetrics:")
print(f"Ground truth triples: {len(ground_graph)}")
print(f"Output triples: {len(output_graph)}")

true_positive = ground_graph & output_graph
false_positive = output_graph - true_positive
false_negative = ground_graph - true_positive

# Save results
results_dir = current_dir / "results" 
try:
    true_positive.serialize(str(results_dir / "true_positive_sentence.ttl"), format="ttl")
    false_positive.serialize(str(results_dir / "false_positive_sentence.ttl"), format="ttl")
    false_negative.serialize(str(results_dir / "false_negative_sentence.ttl"), format="ttl")
    
    print(f"\\nTrue positives: {len(true_positive)}")
    print(f"False positives: {len(false_positive)}")
    print(f"False negatives: {len(false_negative)}")
except Exception as e:
    print(f"Error saving result files: {e}")
    raise

# Calculate final metrics
precision = len(true_positive) / len(output_graph) if len(output_graph) > 0 else 0
recall = len(true_positive) / len(ground_graph) if len(ground_graph) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print("\\nFinal Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# %%



