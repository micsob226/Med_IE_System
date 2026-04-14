import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def main():
    dataset = load_dataset("bigbio/bc5cdr")
    print(dataset)

    example = dataset["train"][0]
    print(json.dumps(example, indent=2))


    chemical_count = 0
    disease_count = 0
    for doc in dataset["train"]:
        for passage in doc["passages"]:
            for entity in passage["entities"]:
                if entity["type"] == "Chemical":
                    chemical_count += 1
                elif entity["type"] == "Disease":
                    disease_count += 1
    print(f"Chemical: {chemical_count}")
    print(f"Disease:  {disease_count}")


    chem_no_mesh = 0
    dis_no_mesh = 0
    for doc in dataset["train"]:
        for passage in doc["passages"]:
            for entity in passage["entities"]:
                if not entity["normalized"]:
                    if entity["type"] == "Chemical":
                        chem_no_mesh += 1
                    elif entity["type"] == "Disease":
                        dis_no_mesh += 1
    print(f"Missing MeSH for Chemical: {chem_no_mesh}")
    print(f"Missing MeSH for Disease:  {dis_no_mesh}")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_counts = []
    for doc in dataset["train"]:
        full_text = ""
        for passage in doc["passages"]:
            full_text += passage["text"] + " "
        token_counts.append(len(tokenizer(full_text)["input_ids"]))

    print(f"Most tokens in a doc: {max(token_counts)}")
    print(f"Mean token count:     {np.mean(token_counts):.1f}")
    print(f"Docs > 512 tokens:    {sum(1 for c in token_counts if c > 512)}")


if __name__ == "__main__":
    main()
