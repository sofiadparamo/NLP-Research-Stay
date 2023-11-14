# Text Preprocessing and Tokenization

## Text Preprocessing in NLP

Text preprocessing transforms raw text into a structured and manageable format. This step is vital for reducing data dimensionality, enhancing data quality, and improving the performance of NLP models.

### Common Techniques:

- **Lowercasing:** Converts all text to lowercase to maintain consistency and reduce complexity.
- **Punctuation Removal:** Eliminates punctuation marks, which are often irrelevant for analysis.
- **Handling Special Characters:** Involves dealing with characters like symbols or emojis that may not contribute to the desired analysis.
- **Eliminating Stop Words:** Removes commonly occurring words like 'and', 'the', etc., that do not add much meaning to the text.

## Tokens:

In both artificial and natural languages, defining what is considered as a token is crucial. While artificial languages allow for precise and unambiguous token definitions, natural languages present a rich variety, making the decision of what constitutes a token more complex​​.

Text, in its raw form, is a sequence of characters. Tokenization involves isolating word-like units from this character stream. These tokens can be punctuation, numbers, dates (structurally recognizable units), or units for morphological analysis​​.

## Preprocessing Text:

Electronic text, often produced as a by-product of typesetting, contains various elements like extra whitespace, font changes, text subdivisions, and special characters. These elements, while meaningful for readers, are usually filtered out in the preprocessing stage, even before tokenization begins​​.

After preprocessing, the text is seen as a string of characters. Linguistic processors then consider elements of this text as belonging to syntactic classes (e.g., the string "dog" as a singular noun). Tokenization plays a critical role in dividing the text into units that are recognized as class members​​.

Besides classifying words, tokenization is also important for identifying sentence boundaries. This is particularly significant as most linguistic analyzers use sentences as their primary unit of treatment​​.

## Tokenization:

- **Tokens:** The smallest units of text, like words or characters. They represent the fundamental elements for textual analysis.
- **Tokenization Process:** Involves breaking down text into individual tokens. It can be at the word level (splitting sentences into words) or character level (splitting into individual characters).
- **Applications:** Essential for various NLP tasks like language modeling, sentiment analysis, and machine translation. Effective tokenization captures the structure of language and aids in efficient text processing.

## Features in Machine Learning for NLP:

Features are measurable attributes extracted from text, serving as input to predictive models. They're crucial in capturing relevant information from the data, influencing predictions in tasks like text classification, named entity recognition, and sentiment analysis.

**Types:**

- Word frequencies
- n-grams (sequences of 'n' words or characters)
- Statistical Representations from tokenized text

## Challenges and Flexibility in Tokenization:

- **Modular Approach:** The tokenization process is suggested to be treated as a series of modular filters, allowing for selective text processing. This approach provides the flexibility to handle various text elements and ambiguities​​.

- **Dealing with Ambiguities:** Tokenization involves resolving character ambiguities and recognizing sentence and word boundaries. The process might also involve rejoining separated parts of proper names or handling spaces as ambiguous separators based on contextual clues, like uppercase letters in English​​.

## Experiment Results

Running the simple tokenization experiment results on two different tokenization level approches:
- By character
- By word

The experiment generates an output where it is possible to visualize each individual token identified in both levels, as well as the parsed text from the chosen website.