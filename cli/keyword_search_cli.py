#!/usr/bin/env python3

import argparse
import json
import string

from nltk.stem import PorterStemmer


stemmer = PorterStemmer()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/stopwords.txt","r") as f:
       stopwords:list[str] = f.read().splitlines()

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

if __name__ == "__main__":
    main()