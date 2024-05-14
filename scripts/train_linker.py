import spacy
from spacy import displacy
import json
from pathlib import Path
import typer
from collections import Counter
import random
from spacy.training import Example
from spacy.ml.models import load_kb
from spacy.util import minibatch, compounding


def main(input_data: Path = typer.Argument(..., dir_okay=False),
         kb_dir:  Path = typer.Argument(..., dir_okay=True),
         test_ratio: float = typer.Argument(...),
         nepoch: int = typer.Argument(...),
         nprint: int = typer.Argument(...)):

    # python train_linker.py "../assets/annotated_text20.jsonl" "../my_output/my_kb" 0.3 30 10  # Accuracy rate: 0.33
    # python train_linker.py "../assets/annotated_text40.jsonl" "../my_output/my_kb" 0.1 100 20 # Accuracy rate: 0.75
    # python train_linker.py "../assets/annotated_text60.jsonl" "../my_output/my_kb" 0.1 70 20  # Accuracy rate: 0.83
    # python train_linker.py "../assets/annotated_text80.jsonl" "../my_output/my_kb" 0.1 60 20  # Accuracy rate: 0.75
    # python train_linker.py "../assets/annotated_text146.jsonl" "../my_output/my_kb" 0.2 50 5 # Accuracy rate: 0.64

    y, dataset = [], []
    with input_data.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            text = example["text"]
            if example["answer"] == "accept":
                QID = example["accept"]
            
                offset = (example["spans"]["start"], example["spans"]["end"])
                entity_label = example["spans"]["label"]
                entities = [(offset[0], offset[1], entity_label)]
                links_dict = {QID: 1.0}   # The QID is always paired with "1" (ie True), meaning the QID is a correct one (ie manually selected).

            dataset.append((text, {"links": {offset: links_dict}, "entities": entities}))


    for text, annot in dataset:
        for span, links_dict in annot["links"].items():
            # (8, 11)
            # {'Q111361019': 1.0}}
            for link, value in links_dict.items():
                # 'Q111361019': 1.0
                if value:
                    y.append(link)
    print(dataset[0], Counter(y))

    random.shuffle(dataset)
    ntest = int(test_ratio * len(dataset))
    test_dataset = dataset[:ntest]
    train_dataset = dataset[ntest:]
    print(len(train_dataset), len(test_dataset))

    nlp = spacy.load("zh_core_web_md")
    TRAIN_EXAMPLES = []
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    sentencizer = nlp.get_pipe("sentencizer")

    for text, annotation in train_dataset:
        example = Example.from_dict(nlp.make_doc(text), annotation)
        example.reference = sentencizer(example.reference)
        TRAIN_EXAMPLES.append(example) 

    entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": False}, last=True)
    entity_linker.initialize(get_examples=lambda: TRAIN_EXAMPLES, kb_loader=load_kb(kb_dir))



    with nlp.select_pipes(enable=["entity_linker"]):   # train only the entity_linker
        optimizer = nlp.resume_training()
    
        for itn in range(nepoch): 
            random.shuffle(TRAIN_EXAMPLES)
            batches = minibatch(TRAIN_EXAMPLES, size=compounding(4.0, 32.0, 1.001))  # increasing batch sizes
            losses = {}
            for batch in batches:
                nlp.update(
                    batch,   
                    drop=0.2,      # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            if itn % nprint == 0:
                print(itn, "Losses", losses)   # print the training loss

    print(itn, "Final losses", losses)

    true_y, predicted_y = [], []
    accuracy_hist_test = [0] * len(test_dataset)

    for i, (text, true_annot) in enumerate(test_dataset):
        doc = nlp(text)  # to make this more efficient, you can use nlp.pipe() just once for all the texts
        for ent in doc.ents:
            if ent.text == "黄国书":
                py = ent.kb_id_
                predicted_y.append(ent.kb_id_)

        for span, links_dict in true_annot["links"].items():
            for link, value in links_dict.items():
                ty = link
                true_y.append(link)

        is_correct = float(py==ty)
        accuracy_hist_test[i] = is_correct

    print("Accuracy rate:", sum(accuracy_hist_test) / len(test_dataset))
    
    acc_file = "../assets/acc.json"
    with open(acc_file, "w") as final:
        json.dump(accuracy_hist_test, final)


if __name__ == "__main__":
    typer.run(main)