import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi


def main():
    dataset = load_dataset("bigbio/bc5cdr")

    # document -> list of words
    tokenized_docs = []
    for doc in dataset["train"]:
        full_text = ""
        for passage in doc["passages"]:
            full_text += passage["text"] + " "
        tokenized_docs.append(full_text.split())

    bm25 = BM25Okapi(tokenized_docs)

    query = "chemical induced hypertension"
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_5 = np.argsort(scores)[::-1][:5]
    print(f"Top 5 documents for query '{query}':")
    print(top_5)


if __name__ == "__main__":
    main()