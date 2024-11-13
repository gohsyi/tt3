import os
import json
import torch
import random
import pathlib
import transformers
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional



class MathDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], tokenizer: Optional[transformers.PreTrainedTokenizer]):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # prompt = MATH_FEWSHOT_PROMPT + GENERATION_PROMPT.format(question=item["problem"])
        prompt = item["problem"]
        inputs = self.tokenizer(prompt) if self.tokenizer is not None else None
        label = item["label"]
        label = self.tokenizer(label)['input_ids'] if self.tokenizer is not None else label
        return prompt, inputs, label

    def sample(self, size: int):
        sampled_indices = random.sample(range(len(self.data)), size)
        sampled_items = [self[i] for i in sampled_indices]
        return sampled_items


def index_processed_math_dataset(
    processed_ds: Dataset,
    splits: List[str] = ["train", "test"],
):
    index = {split: {} for split in splits}
    for split in splits:
        for d in processed_ds[split]:
            index[split][d["problem"]] = d["solution"]
    return index

    
def load_json_file(file: pathlib.Path) -> Dict:
    with open(file, "r") as f:
        return json.load(f)


def math_dataset_provider(
    path: str = "datasets/MATH",
    splits: List[str] = ["train", "test"],
) -> Dict[str, List]:
    ds = {s: list() for s in splits}
    processed_ds = load_dataset("gohsyi/math")
    processed_index = index_processed_math_dataset(processed_ds, splits)
    datasets = {split: list() for split in splits}
    for split in splits:
        split_path = os.path.join(path, split)
        fields = [f for f in pathlib.Path(split_path).iterdir() if f.is_dir()]

        # Iterate through each field and process the JSON files in parallel
        with ThreadPoolExecutor() as executor:
            for field in fields:
                json_files = [file for file in pathlib.Path(field).iterdir() if file.suffix == ".json"]
                
                # Load all JSON files concurrently
                json_data_list = list(executor.map(load_json_file, json_files))
                
                for data in json_data_list:
                    # Find the corresponding solution using the index
                    if data["problem"] in processed_index[split]:
                        data["answer"] = processed_index[split][data["problem"]]
                        # remove "level", "type" keys from the dict
                        data.pop("level", None)
                        data.pop("type", None)
                        data.pop("solution", None)
                        ds[split].append(data)
                    else:
                        raise ValueError(f"Could not find the corresponding data in the processed dataset for problem: {data['problem']}")
        
        data_dict = {
            "question": [d["problem"] for d in ds[split]],
            "answer": [d["answer"] for d in ds[split]]
        }
        datasets[split] = Dataset.from_dict(data_dict)
    return datasets

    
if __name__ == '__main__':
    dataset = math_dataset_provider(path="MATH", tokenizer=None)
    