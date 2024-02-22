from typing import List, Dict


def format_wordnet_dataset(wordnet_dataset: List[Dict], example_required: bool = False) -> List:
    """
    To convert wordnet dataset to concept format, which include `synset`,
    `word`, `description`, `example` columns.
    """
    dataset = []
    for data in wordnet_dataset:
        concept = {
            "synset": data["synset"],
            "word": data["word"].replace('_', ' '),
            "description": data["description"],
            "synonyms": [s.replace('_', ' ') for s in data["lemmas"]],
            "example": '' if not data["examples"] else data["examples"][0]
        }
        if example_required and not concept["example"]:
            continue
        dataset.append(concept)
    return dataset
