import os
import numpy as np
from annoy import AnnoyIndex
import openai
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse

# Set OpenAI API key if using OpenAI
# openai.api_key = 'your-api-key'

def read_text_files(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                documents[file_path] = content
    return documents

def chunk_text(text, chunk_size=100):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

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

def create_annoy_index(directory, index_file, embedding_type, api_key=None, chunk_size=100, batch_size=10):
    documents = read_text_files(directory)
    chunks = []
    doc_chunks = []
    
    for file_path, content in tqdm(documents.items(), desc="Reading and Chunking Documents"):
        file_chunks = chunk_text(content, chunk_size)
        for idx, chunk in enumerate(file_chunks):
            chunks.append(chunk)
            doc_chunks.append((file_path, idx))

    if embedding_type == "openai":
        openai.api_key = api_key
        embedding_size = 1536
    elif embedding_type == "bert":
        embedding_size = 768
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()
    else:
        raise ValueError("Unsupported embedding type. Choose either 'openai' or 'bert'.")

    index = AnnoyIndex(embedding_size, 'euclidean')
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Chunks"):
        batch_chunks = chunks[i:i + batch_size]
        batch_doc_chunks = doc_chunks[i:i + batch_size]
        
        if embedding_type == "openai":
            batch_embeddings = get_embeddings_openai(batch_chunks)
        elif embedding_type == "bert":
            batch_embeddings = get_embeddings_bert(batch_chunks, model, tokenizer, device)
        
        for j, embedding in enumerate(batch_embeddings):
            index.add_item(i + j, embedding)
    
    index.build(10)  # 10 trees
    index.save(index_file)
    
    with open(f'{index_file}.txt', 'w') as f:
        for doc_id, chunk_index in doc_chunks:
            f.write(f"{doc_id},{chunk_index}\n")
    
    print(f"Annoy index created and saved as {index_file}")
    print(f"Document mapping saved as {index_file}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Annoy Index')
    parser.add_argument('directory', type=str, help='Directory containing text files')
    parser.add_argument('index_file', type=str, help='Output index file')
    parser.add_argument('--embedding', type=str, choices=['openai', 'bert'], required=True, help='Embedding type to use')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required if using OpenAI embeddings)')
    args = parser.parse_args()
    
    create_annoy_index(args.directory, args.index_file, args.embedding, args.api_key)
