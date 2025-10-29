# text-to-kg-BE
Documentation of a text-to-knowledge graph pipeline using an LLM (Llama-3.1-8B-Instruct deployed though Hugging Face) for ontology population, and its related evaluation strategy.

The evaluation of this experiment are presented in Boscariol M, Meschini S, Tagliabue LC (2025), "Knowledge engineering with LLMs for asset information management in the built environment". Proceedings of the Institution of Civil Engineers - Smart Infrastructure and Construction https://doi.org/10.1680/jsmic.24.00035

This project explores how Large Language Models (LLMs) can automate the creation of Knowledge Graphs (KGs) from unstructured documents in the built environment domain. It introduces a pipeline that extracts entities and relationships using guided few-shot prompting, outputs Terse RDF Triple Language (Turtle) data, and aligns results with existing domain ontologies.
The approach compares inference at different text granularities (full text, paragraph, and sentence) to evaluate semantic alignment and syntactic consistency. 
The experiment was conducted on a use case from the University of Torino, involving requirements for the renovation of a museum.
Early results show the method’s potential to support asset information management, improve data structuring, and advance knowledge engineering in construction and facility management contexts.



## Project Structure

```
.
├── python scripts llama/         # Main LLM-based text-to-KG scripts
│   ├── llama-text-to-KG-*.py    # Three different granularity approaches
│   └── error analysis/          # Scripts for analyzing and evaluating results
│       └── tables/              # Output tables from analysis
├── input/                       # Input data directory
└── results/                     # Output RDF files directory
```

## Scripts Description and Workflow

### Main Text-to-KG Processing Scripts

Located in `python scripts llama/`:

1. `llama-text-to-KG-full-text.py`: Processes the entire text as a single input
2. `llama-text-to-KG-paragraphs.py`: Processes the text paragraph by paragraph
3. `llama-text-to-KG-sentence.py`: Processes the text sentence by sentence

These scripts take input from the `input/` directory and generate RDF triples in the `results/` directory.

### Error Analysis Scripts

Located in `python scripts llama/error analysis/`. Run these scripts in the following order:

1. `01_ttl-json-df.py`: Converts TTL files to JSON format for analysis
2. `02_gt-json-df.py`: Processes ground truth data into comparable format
3. `03_fp-gt-similarity.py`: Analyzes similarity between false positives and ground truth
4. `03_fp-vs-gt.py`: Compares false positives against ground truth data
5. `04_tables.py`: Generates final analysis tables in the `tables/` directory 

## Input and Output Files

### Input Directory
- `example.txt`: Example input text file, to be fed to the system with a few-shot prompting strategy
- `input.txt`: Main input text file, used to run the experiment
- `ground1.ttl`: Ground truth RDF data for evaluation, referring to the content of `input.txt`

### Results Directory
Contains the output files for each granularity approach (full text, paragraph, sentence):
- Complete RDF output (`rdf_output_*.ttl`)
And all related metrics, per each inference scenario:
- True positives (`true_positive_*.ttl`)
- False positives (`false_positive_*.ttl`)
- False negatives (`false_negative_*.ttl`)

## Case Study Files

The repository currently contains the input files and output results related to the case study of the experiment. The `input/` directory includes the example and input texts used in the study, along with their ground truth RDF data. The `results/` directory and `error analysis/tables/` contain all the generated outputs, evaluation metrics, and analysis tables from this specific experiment.

## Setup and Usage

1. Place your input text in `input/input.txt`
2. Run one of the text-to-KG scripts:
   ```bash
   python "python scripts llama/llama-text-to-KG-full-text.py"
   # or
   python "python scripts llama/llama-text-to-KG-paragraphs.py"
   # or
   python "python scripts llama/llama-text-to-KG-sentence.py"
   ```
3. Run the error analysis scripts in sequence:
   ```bash
   cd "python scripts llama/error analysis"
   python 01_ttl-json-df.py
   python 02_gt-json-df.py
   python 03_fp-gt-similarity.py
   python 03_fp-vs-gt.py
   python 04_tables.py
   ```

The final analysis results will be available in the `error analysis/tables/` directory.

