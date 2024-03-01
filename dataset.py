import pandas as pd
from datasets import Dataset

def create_dataset(
    load_file:str="data/ascii_art_data.json",
    save_file:str="data/ascii_art_dataset"
):
    df = pd.read_json(load_file)
    prompts = []
    for i, row in df.iterrows():
        prompt = f"Topic: {row.topic}\n"
        size = row.pre_suffix
        if size:
            prompt += f"Size: {size}\n"
        prompt += "Art:"
        prompts.append(prompt)
    # df['input'] = prompts
    # df['output'] = df['text']

    prompt = f"<|im_start|> Topic: {df['topic']} Size: {df['pre_suffix']} Art:"
    prompt += df['text'] + "<|im_end|>"

    df = df.drop(columns=['pre_suffix', 'topic'])

    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(save_file)

def load_ascii_dataset():
    dataset = Dataset.load_from_disk("data/ascii_art_dataset")
    return dataset