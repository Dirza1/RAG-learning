#!/usr/bin/env python3

import argparse
import json
from lib.semantic_search import verify_model, SemanticSearch,embed_text,verify_embeddings,embed_query_test

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify",help="Verify the used model")

    embed_parser = subparser.add_parser("embed_text",help="Embed the text provided")
    embed_parser.add_argument("text",type=str, help="The text to embed")

    embeddings_parser = subparser.add_parser("verify_embeddings",help="Build or verrify the embeddings of the movies")

    embedquery_parser = subparser.add_parser("embedquery",help="Embed the query given")
    embedquery_parser.add_argument("query",type=str,help="Querry to embed")

    search_parser = subparser.add_parser("search",help="Search the database with a querry")
    search_parser.add_argument("query",type=str,help="The querry to look for")
    search_parser.add_argument("--limit",type=int,default=5,help="The ammount of results to display")

    args = parser.parse_args()

    match args.command:
        case"verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_test(args.query)
        case "search":
            ss:SemanticSearch = SemanticSearch()
            with open("data/movies.json","r") as file:
                documents = json.load(file)['movies']
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query,args.limit)
            for index,result in enumerate(results):
                print(f"{index + 1}. {result['title']} (score: {result['score']:.4f})")
                print(f"{result['description']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()