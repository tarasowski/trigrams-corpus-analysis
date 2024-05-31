import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def generate_ngrams(tokens, n):
    return [' '.join(grams) for grams in ngrams(tokens, n)]

# Directory containing the content files for each website
content_dir = "test_corpus"

# Read and preprocess the content from each file
preprocessed_contents = {}
for filename in os.listdir(content_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(content_dir, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            preprocessed_contents[filename] = preprocess(content)

# Generate n-grams for each website
ngrams_contents = {}
for filename, tokens in preprocessed_contents.items():
    unigrams = generate_ngrams(tokens, 1)
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    ngrams_contents[filename] = set(unigrams + bigrams + trigrams)

# Convert lists of n-grams into sets for each website
keyword_sets = {filename: ngrams for filename, ngrams in ngrams_contents.items()}

# Find the intersection of n-grams across all websites
if keyword_sets:
    common_ngrams = set.intersection(*keyword_sets.values())
else:
    common_ngrams = set()

# Count occurrences of each common n-gram
keyword_frequency = Counter()
for ngram in common_ngrams:
    for filename, ngrams in ngrams_contents.items():
        if ngram in ngrams:
            keyword_frequency[ngram] += 1

# Print the common n-grams and their frequencies
print("Common N-grams Across All Websites:")
for ngram, frequency in keyword_frequency.items():
    print(f"{ngram}: {frequency}")

