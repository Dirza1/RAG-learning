#!/usr/bin/env python3

import argparse
import json
from lib.semantic_search import verify_model, SemanticSearch,embed_text,verify_embeddings,embed_query_test
import regex as re

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

    chunk_parser = subparser.add_parser("chunk",help="Chunk a piece of text based on a limit")
    chunk_parser.add_argument("text",type=str,help="The text to chunk")
    chunk_parser.add_argument("--chunk-size",type=int,default=200,help="The chunk size to use")
    chunk_parser.add_argument("--overlap",type=int,help="amount of overlap")

    semantic_chunk_parser = subparser.add_parser("semantic_chunk",help="Semanticaly chunk text")
    semantic_chunk_parser.add_argument("text",type=str,help="Text to semanticaly chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size",type=int,default=4,help="How many chunks to make")
    semantic_chunk_parser.add_argument("--overlap",type=int,default=0,help="The ammount of overlap to use")

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
        case "chunk":
            
            split_text:list[str] = args.text.split()
            limit = args.chunk_size
            overlap = args.overlap
            print(f"Chunking {len(args.text)} characters")
            count:int = 1
            while True:
                if len(split_text) <= limit:
                    print(f"{count}. {' '.join(split_text)}")
                    break
                print(f"{count}. {' '.join(split_text[:limit])}")
                split_text = split_text[limit - overlap:]
                count += 1
        case "semantic_chunk":
            text:list[str] = re.split(pattern=r"(?<=[.!?])\s+",string=args.text)
            max_size = args.max_chunk_size
            overlap = args.overlap
            result = semantic_chunking(text,max_size,overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            count = 1
            for itm in result:
                print(f"{count}. {itm}")
            
        case _:
            parser.print_help()

def semantic_chunking(text,limit,overlap)-> list[str]:
    results = []
    while True:
        if len(text) <= limit:
            results.append(f"{' '.join(text)}")
            return results

        results.append(f"{' '.join(text[:limit])}")
        text = text[limit-overlap:]


if __name__ == "__main__":
    main()