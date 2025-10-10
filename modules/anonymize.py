import spacy
import re

# Deutsches Modell laden
nlp = spacy.load("de_core_news_lg")

def anonymize_text(text):
    doc = nlp(text)
    anonymized = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PER", "LOC", "ORG", "GPE"]:
            anonymized = anonymized[:ent.start_char] + f"[{ent.label_}]" + anonymized[ent.end_char:]
    return anonymized

def anonymize_patterns(text):
    text = re.sub(r"\b\d{2,4}[-/]\d{2,4}[-/]\d{2,4}\b", "[DATUM]", text)
    text = re.sub(r"\b\d{3,}\b", "[ZAHL]", text)
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    text = re.sub(r"\b\d{2,5}[-\s]?\d{3,}\b", "[TELEFON]", text)
    return text

def anonymize_full(text):
    text = anonymize_patterns(text)
    text = anonymize_text(text)
    return text