import os
import re
import requests
from bs4 import BeautifulSoup


def download_and_save_text(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text


def word_tokenize(text):
    # Split the text into individual words
    words = text.split()
    return words


def character_tokenize(text):
    # Split the text into individual characters
    characters = list(text)
    return characters


if __name__ == "__main__":
    url = input("Enter the URL of the webpage to parse: ")
    filename = "parsed_text.txt"

    # Download the web page content and save it to a text file
    # you can try with https://www.eluniversal.com.mx/
    download_and_save_text(url, filename)

    # Read the contents of the file
    with open(filename, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Text Preprocessing
    preprocessed_text = preprocess_text(input_text)
    print("Preprocessed Text:", preprocessed_text)

    # Word Tokenization
    words = word_tokenize(preprocessed_text)
    print("Word Tokens:", words)

    # Character Tokenization
    characters = character_tokenize(preprocessed_text)
    print("Character Tokens:", characters)