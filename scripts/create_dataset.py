import spacy
from spacy import displacy
import json
from pathlib import Path
import typer
import chinese_converter


def main(input_data: Path = typer.Argument(..., dir_okay=False),
         output_data: Path = typer.Argument(..., dir_okay=False),
         split_point: int = typer.Argument(...)):

    # python create_dataset.py "../assets/input_text20.txt" "../assets/annotated_text20.jsonl" 10
    # python create_dataset.py "../assets/input_text40.txt" "../assets/annotated_text40.jsonl" 20
    # python create_dataset.py "../assets/input_text60.txt" "../assets/annotated_text60.jsonl" 30
    # python create_dataset.py "../assets/input_text80.txt" "../assets/annotated_text80.jsonl" 40
    # python create_dataset.py "../assets/input_text140.txt" "../assets/annotated_text140.jsonl" 70
    nlp = spacy.load("zh_core_web_md")
    X_train, starts, ends, labels, texts = [], [], [], [], []
    with open(input_data, 'r', encoding='utf-8') as dataset:
        for i, line in enumerate(dataset):
            line = line.strip('\n')
            text = chinese_converter.to_simplified(line)
            X_train.append(text)

            doc = nlp(text)
            for ent in doc.ents:
               if ent.text == '黄国书':
                    texts.append(ent.text)
                    starts.append(ent.start_char)
                    ends.append(ent.end_char)
                    labels.append(ent.label_)

    print(len(texts), len(starts), len(ends), len(labels))


    spans_keys = ['start', 'end', 'text', 'rank', 'label']
    rank = 0
    span_list = []

    for i in range(len(X_train)):
        start = starts[i]
        end = ends[i]
        text = texts[i]
        label = labels[i]
        spans_values = [start, end, text, rank, label]
        output = {}

        for j in range(len(spans_values)):
            output.update({spans_keys[j]: spans_values[j]})
        
        span_list.append(output)

    print(span_list[:3])

    keys = ['text', 'spans', 'accept', 'answer']
    jsonl_output = []

    for i, line in enumerate(X_train):
        if i <split_point:
            qid = 'Q111361019'
        else :
            qid = 'Q19058548'

        text_ = chinese_converter.to_simplified(line)
        values = [text_, span_list[i], qid, "accept"]

        tmp_output = {}
        for j in range(len(keys)):
            tmp_output.update({keys[j]: values[j]})
 
        jsonl_output.append(tmp_output)



    # Write each list item as a separate JSON object in the file
    with open(output_data, "w") as jsonl_file:
        for item in jsonl_output:
            json.dump(item, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")  # Add a newline after each object

if __name__ == "__main__":
    typer.run(main)