import torch
from torch.utils.data import Dataset


# Define dataset class
class NERDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, labels = self.data[idx]

        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        # Extract the label from the 'entities' key
        label_entities = labels.get('entities', [])

        # Create a binary tensor to indicate entity positions
        label_tensor = torch.zeros(inputs['input_ids'].shape[1], dtype=torch.long)
        for entity_start, entity_end, entity_type in label_entities:
            label_tensor[entity_start:entity_end + 1] = 1  # Set the positions of entities to 1

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": label_tensor,
        }
