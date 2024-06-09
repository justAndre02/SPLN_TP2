from gensim.utils import tokenize
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
import nltk
import psycopg2
from transformers import pipeline

stopwords = nltk.corpus.stopwords.words('portuguese')

# Connect to the database and fetch documents
conn = psycopg2.connect(database = "diariorepublica", 
                        user = "postgres", 
                        host= 'localhost',
                        password = "1597535",
                        port = 5432)

cur = conn.cursor()

def preprocess(line):
    line = line.lower()
    tokens = tokenize(line)
    tokens = [token for token in tokens if token not in stopwords]
    return list(tokens)

cur.execute("SELECT * FROM public.dreapp_document;")

records = cur.fetchall()

# Preprocess the notes and questions
notes = [preprocess(record[10]) for record in records] 

# Create a dictionary representation of the documents
dictionary = Dictionary(notes)

# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) tuples
corpus = [dictionary.doc2bow(note) for note in notes]

# Initialize the TF-IDF model
tfidf = TfidfModel(corpus, normalize=True)

# Initialize the answer generation model
generator = pipeline('question-answering', model='neuralmind/bert-base-portuguese-cased')

def answer_question(question):
    # Preprocess the question and convert it to a vector
    question = preprocess(question)
    question_bow = dictionary.doc2bow(question)

    # Calculate the cosine similarity between the question and each notes
    index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
    similarities = index[tfidf[question_bow]]

    # Get the index of the most similar notes
    most_similar_index = similarities.argmax()

    # Fetch the full text of the most similar document
    cur.execute(f"SELECT html_text FROM public.dreapp_documenttext WHERE document_id = {records[most_similar_index][0]}")
    full_text = cur.fetchone()[0]

    # Generate an answer based on the full text
    answer = generator(question=question, context=full_text)

    return answer

# Now you can use the `answer_question` function to get an answer to a question
answer = answer_question("A Nicarágua efetouy o quê?")
print(answer)