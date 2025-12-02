#modules/anonymize.py

import spacy
import re
import sys

# Lade deutsches Modell einmalig
nlp = spacy.load("de_core_news_lg")

def anonymize_text(text):
    doc = nlp(text)
    anonymized = text

    # Personen, Orte, Organisationen anonymisieren
    for ent in reversed(doc.ents):
        if ent.label_ in ["PER", "LOC", "ORG"]:
            anonymized = anonymized[:ent.start_char] + f"[{ent.label_}]" + anonymized[ent.end_char:]

    # Telefonnummern, E-Mails, Kontakte (optional)
    #anonymized = re.sub(r"\b\d{3,}\b", "[KONTAKT]", anonymized)
    #anonymized = re.sub(r"\b[\w\.-]+@[\w\.-]+\b", "[EMAIL]", anonymized)
    anonymized = re.sub(r"\b\d{2,4}[-/]\d{2,4}[-/]\d{2,4}\b", "[DATUM]", anonymized)
    anonymized = re.sub(r"\b\d{3,}\b", "[ZAHL]", anonymized)
    anonymized = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", anonymized)
    anonymized = re.sub(r"\b\d{2,5}[-\s]?\d{3,}\b", "[TELEFON]", anonymized)

    return anonymized
