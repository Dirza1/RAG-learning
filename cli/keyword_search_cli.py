#!/usr/bin/env python3

import argparse
import json
import string
import pickle
import os

from nltk.stem import PorterStemmer
from collections import defaultdict


stemmer = PorterStemmer()
with open("data/stopwords.txt","r") as f:
       stopwords:list[str] = f.read().splitlines()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build the invertedIndex")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            films:dict = load_movies()
            results:list = []
            query:list[str] = transform_text(args.query,stopwords)
            for movie in films["movies"]:
                title_token = transform_text(movie["title"],stopwords)
                if has_match(query,title_token):
                    results.append(movie)
            results.sort(key=lambda movie:movie["id"])
            for i, movie in enumerate(results[:5],1):
                print(f"{i}. {movie['title']}")
        case "build":
            docs:dict = load_movies()
            index = InvertedIndex(docs)
            index.build()
            index.save()
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

class InvertedIndex():
    def __init__(self,docs) -> None:
        self.docs = docs
        self.index:defaultdict[str,set[int]] = defaultdict(set)
        self.docmap:defaultdict = defaultdict()

    def __add_document(self,doc_id:int, text:str) -> None:
        token_text:list[str] = transform_text(text,stopwords)
        for token in token_text:
            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> list[int]:
        # Transform the search term using the same logic as the indexing
        transformed = transform_text(term, stopwords)
        if not transformed:
            return []
    
        # Use the first (and likely only) token from the transformation
        search_token = transformed[0]
        
        # Use .get() to avoid creating empty entries in your defaultdict
        doc_ids = self.index.get(search_token, set())
        return sorted(list(doc_ids))
        
    
    def build(self) -> None:

        for movie in self.docs["movies"]:
            self.__add_document(movie["id"],f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        if not os.path.exists("cache/"):
            os.makedirs("cache/")
        with open("cache/index.pkl","wb") as f:
            pickle.dump(self.index,f)
        with open("cache/docmap.pkl","wb") as f:
            pickle.dump(self.docmap,f)
        


if __name__ == "__main__":
    main()