from sentence_transformers import SentenceTransformer
import numpy as np
import regex as re
from collections import defaultdict

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
        self.document_map = {}
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        chunks_list:list[str] = []
        metadata_chunks:list[dict] = []

        for doc_idx, document in enumerate(documents):
            if document['description'].strip() == "":
                continue
            chunks = semantic_chunking(re.split(pattern=r"(?<=[.!?])\s+",string=document['description']),4,1)
            for idx, chunk in enumerate(chunks):
                chunks_list.append(chunk)
                metadata_chunks.append({
                    "movie_idx":doc_idx,
                    "chunk_idx":idx,
                    "total_chunks":len(chunks),
                })

        self.chunk_embeddings = self.model.encode(chunks_list,show_progress_bar=True)
        self.chunk_metadata = metadata_chunks
        np.save("cache/chunk_embeddings.npy",self.chunk_embeddings)
        with open("cache/chunk_metadata.json","w") as f:
            json.dump({"chunks": metadata_chunks, "total_chunks": len(chunks_list)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self,documents:list[dict]):
        self.document_map = {}
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json","r") as f:
                chunks = json.load(f)
            self.chunk_metadata = chunks['chunks']
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents=documents)
    
    def search_chunks(self,query:str,limit:int = 10) -> list:
        embeding_querry = self.generate_embedding(query)
        chunk_score:list = []
        for idx, chunk in enumerate(self.chunk_embeddings): #type:ignore
            simularity = cosine_similarity(embeding_querry,chunk)
            chunk_score.append(
                {
                    "chunk_idx":self.chunk_metadata[idx]["chunk_idx"],#type:ignore
                    "movie_idx":self.chunk_metadata[idx]["movie_idx"],#type:ignore
                    "score":simularity
                }
            )
        movie_to_score:defaultdict = defaultdict(float)
        for chunk in chunk_score:
            if chunk["movie_idx"] not in movie_to_score:
                movie_to_score[chunk['movie_idx']] = chunk['score']
            elif movie_to_score[chunk['movie_idx']] < chunk['score']:
                movie_to_score[chunk['movie_idx']] = chunk['score']
        
        sorted_movies = sorted(movie_to_score.items(), key=lambda item: item[1],reverse=True)[:limit]
        final_list:list[dict] = []
        for movie in sorted_movies:
            doc = self.documents[movie[0]]
            final_list.append(
                {
                    "id": doc['id'],
                    "title":doc['title'],
                    "document":doc['description'][:100],
                    "score":round(movie[1],4),
                    "metadata":doc.get("metadata",{})
                }
            )
        return final_list


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

def semantic_chunking(text,limit,overlap)-> list[str]:
    results = []
    while True:
        if len(text) <= limit:
            results.append(f"{' '.join(text)}")
            return results

        results.append(f"{' '.join(text[:limit])}")
        text = text[limit-overlap:]