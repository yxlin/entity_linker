import csv
from pathlib import Path

import spacy
from spacy.kb import InMemoryLookupKB
import chinese_converter

import typer
import os


def load_entities(entities_loc):

    names = dict()
    names_aliases = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            name_alias = row[4]
            
            names[qid] = name
            descriptions[qid] = desc
            names_aliases[qid] = name_alias
    return names, descriptions, names_aliases


def main(entity_file: Path = typer.Argument(..., dir_okay=False),
         output_dir: Path = typer.Argument(..., dir_okay=True),):


    name_dict, desc_dict, alias_dict = load_entities(entity_file)

    for QID in name_dict.keys():
        print(f"{QID}, name={name_dict[QID]}, desc={desc_dict[QID]}, alias={alias_dict[QID]}")


    nlp = spacy.load("zh_core_web_md")

    # KB -----------------------------------------------------------
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
    for qid, desc in desc_dict.items():
        text = chinese_converter.to_simplified(desc)
        desc_doc = nlp(text)
        desc_enc = desc_doc.vector
        # 342 is an arbitrary value here
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   


    for qid, name in alias_dict.items():
        # 100% prior probability P(entity|alias)
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])    


    qids = name_dict.keys()  
    probs = [0.5 for qid in qids] # [0.5, 0.5]
    kb.add_alias(alias="黄国书", entities=qids, probabilities=probs)  


    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print(f"Candidates for '老黄国书': {[c.entity_ for c in kb.get_alias_candidates('老黄国书')]}")
    print(f"Candidates for '黄国书': {[c.entity_ for c in kb.get_alias_candidates('黄国书')]}")
    print(f"Candidates for '王成章': {[c.entity_ for c in kb.get_alias_candidates('王成章')]}")

    # change the directory and file names to whatever you like
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 
    
    kb.to_disk(output_dir / "my_kb")
    nlp.to_disk(output_dir / "my_nlp")

if __name__ == "__main__":
    typer.run(main)
# python preprocess.py "../assets/entities.csv" "../my_output/"