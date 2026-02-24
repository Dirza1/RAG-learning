from sentence_transformers import SentenceTransformer
import numpy as np

import os
import json

class SemanticSearch:
    def __init__(self) -> None:
        self.model:SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self,text:str):
        if text.strip() == "" or not text:
            raise ValueError("Text nly contains white space")
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self,documents):
        self.documents = documents
        document_list:list[str] = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(document_list,show_progress_bar=True)
        np.save("cache/movie_embeddings.npy",self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self,documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model()->None:
    ss:SemanticSearch = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text:str) -> None:
    ss:SemanticSearch = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss:SemanticSearch = SemanticSearch()
    with open("data/movies.json","r") as fp:
        documents = json.load(fp)['movies']
    embeddings = ss.load_or_create_embeddings(documents=documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_test(query:str) -> None:
    ss:SemanticSearch = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")