from sentence_transformers import SentenceTransformer
import numpy as np

import os
import json

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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
    
    def search(self,query,limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embed_query = self.generate_embedding(query)
        simalarity_list:list[tuple] = []
        for index ,embedding in enumerate(self.embeddings):
            simularity = cosine_similarity(embed_query,embedding)
            simalarity_list.append((simularity,self.documents[index])) #type:ignore
        sorted_sim = sorted(simalarity_list,key=lambda x: x[0],reverse=True)
        return_list = sorted_sim[:limit]
        final_return:list[dict] = []
        for result in return_list:
            final_return.append({
                "score" : result[0],
                "title" : result[1]["title"],
                "description" : result[1]["description"],
            })
        return final_return

        
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self,documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        chunks:list[str] = []
        metadata_chunks:dict = {}


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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)