import json
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess a document
def preprocess(document):
    stopwords_set = set(stopwords.words('portuguese'))
    words = word_tokenize(document.lower())
    words = [word for word in words if word not in stopwords_set]
    return words

# Load all the documents
documents = []
original_documents = []
for i in range(20):
    with open(f'documentos_part_{i+1}.json', 'r', encoding='utf-8') as f:
        print(f'Loading part {i+1}...')
        part = json.load(f)
        original_documents.extend(part)
        for doc in part:
            documents.append(preprocess(doc['notes']))

# Create a dictionary and a corpus for the TF-IDF model
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(document) for document in documents]

# Train the TF-IDF model
tfidf = TfidfModel(corpus)

# Preprocess the input topic and convert it to a bag-of-words format
topic = input('Introduz o tópico da tua questão: ')
topic_bow = dictionary.doc2bow(preprocess(topic))

# Use the TF-IDF model to convert the bag
# -of-words topic into a TF-IDF vector
topic_tfidf = tfidf[topic_bow]

# Calculate the cosine similarity between the TF-IDF vector of the topic and the TF-IDF vectors of the documents
index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
sims = index[topic_tfidf]

# Sort the documents by their similarity to the topic and select the top 200
top_200_indices = sorted(range(len(sims)), key=lambda i: -sims[i])[:200]
top_200_documents = [original_documents[i] for i in top_200_indices]

# Write the selected entries to a new JSON file
with open('snippet.json', 'w', encoding='utf-8') as f:
    json.dump(top_200_documents, f, ensure_ascii=False, indent=4)
