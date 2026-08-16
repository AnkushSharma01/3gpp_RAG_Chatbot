import json
import os

ACRONYM_PATH = os.path.join("data", "3gpp_acronyms.json")

def expand_telecom_query(query: str) -> str:
    if not os.path.exists(ACRONYM_PATH):
        return query

    with open(ACRONYM_PATH, "r") as f:
        acronyms = json.load(f)

    words = query.split()
    expanded_words = []
    for word in words:
        clean_word = word.strip(",.?()").upper()
        if clean_word in acronyms:
            expanded_words.append(f"{word} ({acronyms[clean_word]})")
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)
