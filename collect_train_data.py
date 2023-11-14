import re

import pandas as pd
import pycountry
import requests
from bs4 import BeautifulSoup

NON_RELEVANT_WORDS = ["web", "guest", "question", "ask", "add", "read", "about", "be", "buy", "call",
                      "chat", "order", "close", "contact", "country", "customer", "default",
                      "delivery", "discount", "email", "enjoy" "enter", "explore", "free",
                      "find", "finish", "free", "help", "how", "if", "in", "usd", "shop", "sign",
                      "sort", "price", "trade", "your", "comment", "from", "to", "you", "view",
                      "twit", "south", "save", "product", "privacy", "accept", "terms", "sold",
                      "pay", "friday", "by", "cult", "cancel", "christmas", "clear", "color",
                      "commercial", "create", "new", "click", "sale", "learn", "all"]

SPECIFIC_SYMBOLS = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"

WORDS_TO_FIND = ["sofa","compressor","fryer","hose","bookcase","bookshelves",
                 "cushion","lamp","pump","dishwasher","shelves",
                 "stairs","mattress","rug","utensil","appliance","dresser","drawer",
                 "mirror","jug","pot","pillow","accessories","sofa","vase",
                 "blanket","chair","furniture","table","decor","mats"]


# country name check
def is_country_name(word):
    try:
        pycountry.countries.lookup(word.lower())
        return True
    except LookupError:
        return False


# Remove leading and trailing spaces and replace multiple spaces with a single space
def correct_spaces(word):
    return re.sub(r'\s+', ' ', word.strip())


# Web scraping function
def scrape_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            return text_content
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def collect_data(urls):
    text = ""
    for url in urls:
        if (url.startswith("http")):
            text_content = scrape_page(url)
            if text_content:
                text += " " + text_content
        else:
            print(f"{url} is not URL")
    return text


def clean_text(text):
    # remove specific symbols
    cleaned_text = re.sub(f"[{re.escape(SPECIFIC_SYMBOLS)}]", "", text)
    # remove numbers
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    # remove extra new lines
    cleaned_text = '\n'.join(sorted(set(line.strip().lower() for line in cleaned_text.split('\n'))))
    print(len(cleaned_text))
    cleaned_data = []
    for word in cleaned_text.split('\n'):
        # remove non alphabet symbols, non relevant words, and country names, save in lower case
        if (all(ord(char) < 128 for char in word) and len(word) > 2 and not is_country_name(word)):
            #if (not any(keyword in word for keyword in NON_RELEVANT_WORDS)):
                cleaned_data.append(correct_spaces(word))
    # remove duplicates and sort it, then return
    cleaned_data = sorted(set(cleaned_data))
    print(len(cleaned_data))
    return cleaned_data

def collect_sentences_with_keywords(sentences, keywords):
    collected_sentences = []

    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            collected_sentences.append(sentence)

    return collected_sentences

def find_word_indexes_in_sentences(sentences, words_to_find):
    result = []

    for sentence in sentences:
        sentence_result = {"sentence": sentence, "entities": []}

        for word in words_to_find:
            word_indexes = []
            start_index = sentence.lower().find(word.lower())

            while start_index != -1:
                end_index = start_index + len(word) #- 1
                word_indexes.append((start_index, end_index, 'PRODUCT'))
                start_index = sentence.lower().find(word.lower(), start_index + 1)

            if word_indexes:
                sentence_result["entities"].extend(word_indexes)

        if sentence_result["entities"]:
            result.append((sentence_result["sentence"], {"entities": sentence_result["entities"]}))

    return result


def get_train_data(path):
    df = pd.read_csv(path, header=None)
    sentences = df.iloc[:, 0].tolist()
    train_data = []
    result = find_word_indexes_in_sentences(sentences, WORDS_TO_FIND)
    for sentence, entities in result:
        train_data.append((sentence, entities))

    return train_data


def collect_and_save_data(urlCsvPath):
    df = pd.read_csv(urlCsvPath)
    urls = df["url"].tolist()

    text = collect_data(urls)
    cleaned_data = clean_text(text)
    collected_sentences = collect_sentences_with_keywords(cleaned_data, WORDS_TO_FIND)
    print(collected_sentences)
    df = pd.DataFrame(collected_sentences)
    df.to_csv("training_data.csv", header=False, index=False)
    print("Done")

csv_file_path = '/Users/mibsib/PycharmProjects/furniture_store/some_furniture stores pages.csv'

csv_collected = "training_data.csv"

train_data =get_train_data(csv_collected)
print(train_data)
import prepare_train_data as prep_data

# df = pd.read_csv("training_data1.csv", header=None)
# train_sentences = df.iloc[:, 0].tolist()
# collected_sentences = collect_sentences_with_keywords(train_sentences, WORD_TO_FIND)
# print(collected_sentences)
# df = pd.DataFrame(collected_sentences)
# df.to_csv("training_data2.csv", header=False, index=False)
#
# quit()

# csv file path
# csv_file_path = '/Users/mibsib/PycharmProjects/furniture_store/some_furniture stores pages.csv'
# df = pd.read_csv(csv_file_path)
# urls = df["url"].tolist()
#
# text = collect_data(urls)
# cleaned_data = clean_text(text)
# # save as training data
# # df = pd.DataFrame(cleaned_data)
# collected_sentences = collect_sentences_with_keywords(cleaned_data, WORDS_TO_FIND)
# print(collected_sentences)
# df = pd.DataFrame(collected_sentences)
# df.to_csv("training_data.csv", header=False, index=False)
# print("Done")

