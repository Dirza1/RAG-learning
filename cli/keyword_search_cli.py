#!/usr/bin/env python3

import argparse
import json
import string
import pickle
import os
import math

from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from constants import BM25_K1, BM25_B


stemmer = PorterStemmer()
with open("data/stopwords.txt","r") as f:
       stopwords:list[str] = f.read().splitlines()

class InvertedIndex():
    def __init__(self) -> None:
        self.index:defaultdict[str,set[int]] = defaultdict(set)
        self.docmap:defaultdict = defaultdict()
        self.term_frequencies:defaultdict[int,Counter] = defaultdict(Counter)
        self.doc_length:defaultdict[int,float] = defaultdict(float)
        self.doc_lengths_path = os.path.join("cache","doc_lengths.pkl")

    def __add_document(self,doc_id:int, text:str) -> None:
        token_text:list[str] = transform_text(text,stopwords)
        for token in token_text:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_length[doc_id] = len(token_text)
    
    def __get_avg_doc_length(self) -> float:
        total_doc = len(self.docmap)
        if total_doc == 0:
            return 0.0
        count = 0
        for _,val in self.doc_length.items():
            count += val
        return count / total_doc
    
    def get_documents(self, term: str) -> list[int]:
        transformed = transform_text(term, stopwords)
        if not transformed:
            return []
    
        search_token = transformed[0]
        doc_ids = self.index.get(search_token, set())
        return sorted(list(doc_ids))
        
    
    def build(self) -> None:
        docs = load_movies()
        for movie in docs["movies"]:
            self.__add_document(movie["id"],f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        if not os.path.exists("cache/"):
            os.makedirs("cache/")
        with open("cache/index.pkl","wb") as f:
            pickle.dump(self.index,f)
        with open("cache/docmap.pkl","wb") as f:
            pickle.dump(self.docmap,f)
        with open("cache/term_frequencies.pkl","wb") as f:
            pickle.dump(self.term_frequencies,f)
        with open(self.doc_lengths_path,"wb") as f:
            pickle.dump(self.doc_length,f)

    def load(self) -> None:
        if not os.path.exists("cache/"):
            raise NotADirectoryError("The directory 'cache' does not exists. Files needs to be saved before they can be loaded")
        try:
            with open("cache/index.pkl","rb") as f:
                index_file = pickle.load(f)
            self.index = index_file
        except FileNotFoundError:
            raise FileExistsError("index file not avalibe")
        
        try:
            with open("cache/docmap.pkl","rb") as f:
                docmap_file = pickle.load(f)
            self.docmap = docmap_file
        except FileNotFoundError:
            raise FileNotFoundError("Docmap file not avalible")
        
        try:
            with open("cache/term_frequencies.pkl","rb") as f:
                term_frequencies_file = pickle.load(f)
            self.term_frequencies = term_frequencies_file
        except FileNotFoundError:
            raise FileNotFoundError("term_frequencies file not avalible")
        
        try:
            with open(self.doc_lengths_path,"rb") as f:
                doc_lenghs = pickle.load(f)
            self.doc_length = doc_lenghs
        except FileNotFoundError:
            raise FileNotFoundError("doc_length file not avalible")
    
    def get_tf(self,doc_id:int, term:str) -> int:
        token = transform_text(term,stopwords)
        if len(token) > 1:
            raise Exception("To many arguments given to the tf command")
        
        return self.term_frequencies[doc_id][token[0]]
    
    def get_idf(self,term:str)->float:
        token = transform_text(term,stopwords)[0]
        total_docs = len(self.term_frequencies)
        docs_with_term = self.get_documents(token)
        return math.log((total_docs+1 ) / (len(docs_with_term) +1))

    def get_bm25_idf(self, term:str) -> float:
        token:list[str] = transform_text(term,stopwords)
        if len(token) > 1:
            raise Exception("Input is more then one token")
        df:int = len(self.get_documents(token[0]))
        n:int = len(self.term_frequencies)
        return math.log((n - df + 0.5) / (df +0.5) +1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1,b=BM25_B) -> float:
        token = transform_text(term,stopwords)
        if len(token) > 1:
            raise Exception("To many arguments pased")
        raw_tf = self.get_tf(doc_id,token[0])
        length_norm:float = 1 - b + b * (self.doc_length[doc_id] / self.__get_avg_doc_length())
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
    
    def bm25(self,doc_id,term) -> float:
        tf = self.get_bm25_tf(doc_id,term,BM25_K1,BM25_B)
        idf = self.get_bm25_idf(term)
        return tf * idf

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build the invertedIndex")

    tf_parser = subparsers.add_parser("tf",help="Generate token frequency")
    tf_parser.add_argument("doc_id",help="The docuemnt ID to check")
    tf_parser.add_argument("term",help="The keyword for wich the counter is to be returned")

    idf_parser = subparsers.add_parser("idf",help="display the inverse index of the argument given")
    idf_parser.add_argument("term",help="the term to be searched for")

    tfidf_parser = subparsers.add_parser("tfidf",help="Generates a tf-idf score for the term prodivded")
    tfidf_parser.add_argument("doc_id",help="The ID of the document searched")
    tfidf_parser.add_argument("term", help="The term to index")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search(index,args.query)
            if not results:
                return
            for result in results:
                print(result)
        case "build":
            index.build()
            index.save()
        case "tf":
            try:
                index.load()
            except Exception as e:
                print(e)
                return
            print(index.get_tf(int(args.doc_id),args.term))
        case "idf":
            token = transform_text(args.term,stopwords)[0]
            index.load()
            print(f"Inverse document frequency of '{args.term}': {index.get_idf(token):.2f}")
            
        case "tfidf":
            try:
                index.load()
            except Exception as e:
                print(e)
                return
            tf_idf = index.get_tf(int(args.doc_id),args.term) * index.get_idf(args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            try:
                index.load()
            except Exception as e:
                print(e)
                return
            bm25_idf:float = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            result = bm25tf(args.doc_id,args.term,index,args.k1,args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {result:.2f}")
        case _:
            parser.print_help()


def load_movies()->dict:
     with open("data/movies.json","r") as fp:
            return json.load(fp)
     
def transform_text(input:str,stopwords:list[str]) ->list[str]:
    translation = str.maketrans("","",string.punctuation)
    output:list[str] = input.translate(translation).lower().split()
    filtered_output:list[str] = []
    for token in output:
        if token not in stopwords:
            filtered_output.append(token)
    
    stem_output = []
    for token in filtered_output:
        stem_output.append(stemmer.stem(token))
    return stem_output

def has_match(query_tokens, title_tokens):
    title_set = set(title_tokens)
    return any(q in title_set for q in query_tokens)

def search(index,term) -> list[str]:
    results:list = []
    try:
        index.load()
    except Exception as e:
        print(e)
        return []
    query:list[str] = transform_text(term,stopwords)
    for token in query:
        ids:list[int] = index.get_documents(token)
        for doc_id in ids:
            if len(results) >= 5:
                break
            else:
                results.append(f"{index.docmap[doc_id]["id"]}: {index.docmap[doc_id]["title"]}")
        if len(results) >= 5:
            break
    return results

def bm25tf(doc_id,term,index:InvertedIndex,k1=BM25_K1,b=BM25_B)-> float:
    index.load()
    return index.get_bm25_tf(doc_id,term,k1,b)

if __name__ == "__main__":
    main()