import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

SPECIFIC_SYMBOLS = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"

WORDS_TO_FIND = ["sofa", "compressor", "fryer", "hose", "bookcase", "bookshelves",
                 "cushion", "lamp", "pump", "dishwasher", "shelves",
                 "stairs", "mattress", "rug", "utensil", "appliance", "dresser", "drawer",
                 "mirror", "jug", "pot", "pillow", "accessories", "sofa", "vase",
                 "blanket", "chair", "furniture", "table", "decor", "mats"]


# Remove leading and trailing spaces and replace multiple spaces with a single space
def correct_spaces(word):
    return re.sub(r'\s+', ' ', word.strip())


# Web scraping function
def scrape_page(url):
    """
      Scrape a web page and extract its text content.

      Parameters:
      - url (str): The URL of the web page to scrape.

      Returns:
      - str or None: Text content of the web page if successful, None otherwise.
      """
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


def quick_scrape_page(url):
    """
       Perform a quick scrape of a web page to extract text content.
       Ignores exeptions

       Parameters:
       - url (str): The URL of the web page to scrape.

       Returns:
       - str: Cleaned text content extracted from the web page.
       """
    text_content = ""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
    except requests.RequestException:
        pass
    finally:
        if text_content != "":
            text_content = clean_input_content(text_content)
        return text_content


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
    cleaned_data = []
    for word in cleaned_text.split('\n'):
        # remove non alphabet symbols, non relevant words, and country names, save in lower case
        if (all(ord(char) < 128 for char in word) and len(word) > 2):
            cleaned_data.append(correct_spaces(word))
    # remove duplicates and sort it, then return
    cleaned_data = sorted(set(cleaned_data))
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
                end_index = start_index + len(word)  # - 1
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
    result = find_word_indexes_in_sentences(cut_sentence(sentences), WORDS_TO_FIND)
    for sentence, entities in result:
        train_data.append((sentence, entities))

    return train_data


def collect_and_save_data(urlCsvPath):
    df = pd.read_csv(urlCsvPath)
    urls = df["url"].tolist()

    text = collect_data(urls)
    cleaned_data = clean_text(text)
    collected_sentences = collect_sentences_with_keywords(cleaned_data, WORDS_TO_FIND)
    # print(collected_sentences)
    df = pd.DataFrame(collected_sentences)
    df.to_csv("training_data.csv", header=False, index=False)
    print("Collect and save done.")


# cut target sentence if it is longer than 255
def cut_sentence(sentence, max_length=255):
    if len(sentence) <= max_length:
        return sentence
    else:
        return sentence[:max_length]


# cut target sentence if it
def clean_input_content(raw_content):
    # print(raw_content)
    # remove specific symbols
    cleaned_text = re.sub(f"[{re.escape(SPECIFIC_SYMBOLS)}]", "", raw_content)
    # remove numbers
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    # remove extra new lines
    cleaned_text = ' '.join(sorted(set(line.strip().lower() for line in cleaned_text.split('\n'))))

    # make it smaller texts to predict
    words = cleaned_text.split()
    ten_word_list = [' '.join(words[i:i + 10]) for i in range(0, len(words) - 9, 10)]
    return ten_word_list


# read urls from given csv file and ignore non url strings
def load_targat_urls(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    urls = df.iloc[:, 0].tolist()
    urls = [url for url in urls if url.startswith('https://')]
    return urls
