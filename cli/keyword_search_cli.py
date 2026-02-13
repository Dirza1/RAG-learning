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
    index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results:list = []
            try:
                index.load()
            except Exception as e:
                print(e)
                return
            query:list[str] = transform_text(args.query,stopwords)
            results:list = []
            for token in query:
                ids:list[int] = index.get_documents(token)
                for doc_id in ids:
                    if len(results) >= 5:
                        break
                    else:
                        results.append(f"{index.docmap[doc_id]["id"]}: {index.docmap[doc_id]["title"]}")
                if len(results) >= 5:
                    break
            for result in results:
                print(result)
        case "build":
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
    def __init__(self) -> None:
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
        


if __name__ == "__main__":
    main()