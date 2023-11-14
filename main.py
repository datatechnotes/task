import os
import data_preparation as prep_data
import model as model
import torch
from transformers import BertForTokenClassification, BertTokenizer


# prepares train data
def prepare_train_data(csv_collected, url_short_list):
    """
    Prepare training data for a machine learning model.

    Parameters:
    - csv_collected (str): Path to the CSV file containing collected data.
    - url_short_list (list): List of URLs to collect data from.

    Returns:
    - train_data (pandas.DataFrame): Training data for the machine learning model.
    """
    if not os.path.exists(csv_collected):
        prep_data.collect_and_save_data(url_short_list)

    train_data = prep_data.get_train_data(csv_collected)
    return train_data


def predict_products(csv_file_path, model_name):
    """
    Predict products in web pages using a trained BERT-based model.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing target URLs.
    - model_name (str): Name or path of the pre-trained BERT model for token classification.

    Note: Assumes the target URLs are stored in the CSV file.

    Prints a message if a product is predicted on a web page.

    Example:
    predict_products("target_urls.csv", "bert-base-uncased")
    """
    urls = prep_data.load_targat_urls(csv_file_path)
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Load the trained model
    model = BertForTokenClassification.from_pretrained(model_name)

    for url in urls:
        predicted_entities = []
        texts_to_predict = prep_data.quick_scrape_page(url)
        if len(texts_to_predict) > 0:
            for text in texts_to_predict:
                inputs = tokenizer.encode_plus(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
                predicted_entities.append([token for i, token in enumerate(tokens) if predictions[i] == 1])

        if (len(predicted_entities) > 0):
            print(f"PRODUCT available in this page! URL: {url}")


if __name__ == '__main__':
    # File paths and names
    url_short_list = 'short furniture stores pages.csv'
    csv_collected = "training_data.csv"
    model_name = "ner_model"

    # Prepare training data
    train_data = prepare_train_data(csv_collected, url_short_list)

    # Train the model if it doesn't exist
    if not os.path.exists(model_name):
        model.train_model(train_data, model_name)

    # File path for URLs to predict
    urls_file_path = 'furniture stores pages.csv'

    # Predict products on web pages
    predict_products(urls_file_path, model_name)
