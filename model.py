import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from tqdm import tqdm
from ner_dataset import NERDataset

# train model with given data
def train_model(train_data, model_name):
    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Assuming 2 labels

    # Prepare the dataset
    dataset = NERDataset(train_data, tokenizer)
    print(dataset.data)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Training loop
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5,weight_decay=1e-4)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_counts = labels.sum(dim=1)
            #print(label_counts)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

    # Save the model
    model.save_pretrained(model_name)

