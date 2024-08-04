# Annoy Search Index for GitHub Repositories with BERT and OpenAI Embeddings

This project provides a tool to search through GitHub repositories efficiently. By creating an index from the contents of repositories, you can quickly find relevant information within large codebases. The tool uses advanced techniques to transform text into searchable vectors, making it easy to locate specific pieces of code or documentation.



## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Creating the Index](#creating-the-index)
- [Searching the Index](#searching-the-index)
- [Usage Examples](#usage-examples)
- [Future Objectives](#future-objectives)


## Overview

This project provides scripts to:
1. Clone github repositories and flatten them ( storing all files from differents levels in 1 directory).
1. Generate text embeddings using either BERT or OpenAI's embedding model.
2. Create an Annoy index with these embeddings.
3. Perform searches on the Annoy index and retrieve relevant text chunks.

## Requirements

- Python 3.7+
- `torch`
- `transformers`
- `openai`
- `annoy`  : https://github.com/spotify/annoy
- `tqdm`

Install the required packages using pip:

```bash
pip install torch transformers openai annoy tqdm
```
## Setup :

```bash 
git clone https://github.com/sinisterplaguebot/annoy-search-index.git

cd annoy-search-index
```
# Usage :

## Clone and Flatten a Github Repo :

```bash
python clone.py https://github.com/{owner}/{repo-name}
```

## Create the index :
```bash
python create_index.py --directory path/to/text/files --embedding [openai|bert] --api_key your-openai-api-key
```
## Searching the index :
```bash
python search_index.py annoy_index.ann "your search query" --embedding [openai|bert] --api_key your-openai-api-key
```
### Note :
the index creation and searching  must use the same embedding .
## Future Objectives :
1- adding an Ai assistant rag powered to simplify the process .
<br>
2- creating a web-app or a google extension for the project .
