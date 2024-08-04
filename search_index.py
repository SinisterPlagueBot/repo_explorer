import os
import numpy as np
from annoy import AnnoyIndex
import openai
import torch
from transformers import BertTokenizer, BertModel
import argparse

# Set OpenAI API key if using OpenAI
# openai.api_key = 'your-api-key'

def get_embeddings_openai(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return np.array(embeddings, dtype=np.float32)

def get_embeddings_bert(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

def load_annoy_index(index_file, embedding_type):
    if embedding_type == "openai":
        embedding_size = 1536
    elif embedding_type == "bert":
        embedding_size = 768
    else:
        raise ValueError("Unsupported embedding type. Choose either 'openai' or 'bert'.")
    
    index = AnnoyIndex(embedding_size, 'euclidean')
    index.load(index_file)
    print(f"Annoy index loaded from {index_file}")
    return index

def load_document_mapping(index_file):
    mapping_file = f'{index_file}.txt'
    doc_chunks = []
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    doc_id, chunk_index = parts
                    doc_chunks.append((doc_id, int(chunk_index)))
        print(f"Document mapping loaded from {mapping_file}")
    else:
        raise FileNotFoundError(f"Document mapping file {mapping_file} not found")
    return doc_chunks

def load_chunks(doc_chunks):
    chunks = {}
    for doc_id, chunk_index in doc_chunks:
        if doc_id not in chunks:
            with open(doc_id, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                chunks[doc_id] = chunk_text(content, chunk_size=100)
    return chunks

def chunk_text(text, chunk_size=100):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def search_annoy_index(query, index, embedding_type, api_key=None, k=5):
    if embedding_type == "openai":
        openai.api_key = api_key
        query_embedding = get_embeddings_openai([query])[0]
    elif embedding_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()
        query_embedding = get_embeddings_bert([query], model, tokenizer, device)[0]
    
    indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)
    
    return indices, distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search Annoy Index')
    parser.add_argument('index_file', type=str, help='Annoy index file')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--embedding', type=str, choices=['openai', 'bert'], required=True, help='Embedding type to use')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required if using OpenAI embeddings)')
    args = parser.parse_args()
    
    index = load_annoy_index(args.index_file, args.embedding)
    doc_chunks = load_document_mapping(args.index_file)
    
    # Load the chunks based on the document mapping
    chunks_dict = load_chunks(doc_chunks)
    
    indices, distances = search_annoy_index(args.query, index, args.embedding, args.api_key)
    
    print("Search Results:")
    for i, idx in enumerate(indices):
        if idx < len(doc_chunks):
            doc_id, chunk_index = doc_chunks[idx]
            chunk_text = chunks_dict[doc_id][chunk_index]
            print(f"Document: {doc_id}, Chunk Index: {chunk_index}, Distance: {distances[i]}")
            print(f"Chunk Text: {chunk_text}\n")
