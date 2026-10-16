"""Unchanged document-local BM25; never receives gold answers."""
import re
from collections import Counter

def words(value):
    return re.findall('[a-z0-9]+', value.lower())

def retrieve(question, texts, top=2):
    tokens = [words(text) for text in texts]
    query = set(words(question))
    average = sum(map(len, tokens)) / max(1, len(tokens))
    import math
    frequency = {word: sum((word in set(text) for text in tokens)) for word in query}
    scores = []
    for (index, text) in enumerate(tokens):
        counts = Counter(text)
        score = 0.0
        for word in query:
            tf = counts[word]
            idf = math.log(1 + (len(tokens) - frequency[word] + 0.5) / (frequency[word] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(text) / max(average, 1)))
        scores.append((score, index))
    return [index for (_, index) in sorted(scores, key=lambda value: (-value[0], value[1]))[:top]]
