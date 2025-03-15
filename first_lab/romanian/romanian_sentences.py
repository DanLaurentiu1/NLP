import spacy
from first_lab.sentences import Sentences
from prettytable import PrettyTable

nlp = spacy.load("ro_core_news_sm")
print("Pipeline:", nlp.pipe_names)

for k, sentence in Sentences.romanian.items():
    table = PrettyTable()
    table.field_names = ["Text", "Lemma", "POS", "POS-Tag", "Syntactic dependency", "Shape", "Is common", "Morphology"]
    table.title = sentence
    doc = nlp(sentence)
    for token in doc:
        table.add_row([token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_stop, token.morph])
    with open(f"../files/romanian_sentences_processing_{k}.txt", "w", encoding="utf-8") as w:
        w.write(str(table))

"""
nlp.add_pipe("merge_noun_chunks")
for k, sentence in Sentences.english.items():
    doc = nlp(sentence)
    chunks = []
    for token in doc:
        chunks.append(token.text)
    with open(f"../files/romanian_sentences_chunking_{k}.txt", "w") as w:
        w.write(str(chunks))
"""
