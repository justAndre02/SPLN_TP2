import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
import psycopg2

class Preprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stopwords = set(stopwords.words('portuguese'))

    def preprocess(self, line):
        line = line.lower()
        tokens = word_tokenize(line)
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def fetch_documents(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            records = json.load(file)
        return records

    def preprocess_notes(self, records):
        notes = [self.preprocess(record['notes']) for record in records]
        return notes

class AnswerExtractor:
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_sm")
        self.nlp.max_length = 10000000 

    def get_named_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.start_char) for ent in doc.ents]
        return entities

class QAPairCreator:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.answer_extractor = AnswerExtractor()

    def create_qa_pairs(self, documents):
        qa_pairs = []

        for doc in documents:
            document_id = doc['id']
            text = doc['notes']

            # Fetch the full text of the document
            conn = psycopg2.connect(database="diariorepublica",
                                    user="postgres",
                                    host='localhost',
                                    password="1597535",
                                    port=5432)
            cur = conn.cursor()
            cur.execute(f"SELECT html_text FROM public.dreapp_documenttext WHERE document_id = {document_id}")
            result = cur.fetchone()
            if result is not None:
                full_text = result[0]
            else:
                full_text = ""

            conn.close()

            # Generate candidate answers using NER
            entities = self.answer_extractor.get_named_entities(full_text)

            # Define questions (in Portuguese)
            questions = [
                "Qual é o tema principal do documento?",
                "Quem é o autor ou a autoridade emissora do documento?",
                "Quando o documento foi publicado?",
                "Quais locais ou regiões são referenciados no documento?",
            ]

            # Match questions with identified entities to create QA pairs
            for question in questions:
                for entity, start_char in entities:
                    qa_pairs.append({
                        "context": full_text,
                        "question": question,
                        "answers": {"text": [entity], "answer_start": [start_char]}
                    })

        return qa_pairs

class QA_System:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.qa_pair_creator = QAPairCreator()
        self.generator = pipeline('question-answering', model='pierreguillou/bert-base-cased-squad-v1.1-portuguese')

    def preprocess_and_create_qa_pairs(self, file_path):
        records = self.preprocessor.fetch_documents(file_path)
        qa_pairs = self.qa_pair_creator.create_qa_pairs(records)
        return qa_pairs

    def answer_question(self, question, file_path):
        # Preprocess the question and convert it to a vector
        question_tokens = self.preprocessor.preprocess(question)
        dictionary = Dictionary(self.preprocessor.preprocess_notes(self.preprocessor.fetch_documents(file_path)))
        question_bow = dictionary.doc2bow(question_tokens)

        # Create a dictionary and TF-IDF model
        corpus = [dictionary.doc2bow(note) for note in self.preprocessor.preprocess_notes(self.preprocessor.fetch_documents(file_path))]
        tfidf = TfidfModel(corpus, normalize=True)

        # Calculate the cosine similarity between the question and each note
        index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
        similarities = index[tfidf[question_bow]]

        # Get the index of the most similar note
        most_similar_index = similarities.argmax()
        records = self.preprocessor.fetch_documents(file_path)
        print(f"Most similar document: {records[most_similar_index]['id']}")

        # Fetch the full text of the most similar document
        conn = psycopg2.connect(database="diariorepublica",
                                user="postgres",
                                host='localhost',
                                password="1597535",
                                port=5432)
        cur = conn.cursor()
        cur.execute(f"SELECT html_text FROM public.dreapp_documenttext WHERE document_id = {records[most_similar_index]['id']}")
        result = cur.fetchone()
        conn.close()

        full_text = result[0] if result else ""

        # Generate an answer based on the full text
        answer = self.generator(question=question, context=full_text)

        return answer

file_path = "snippet.json"
qa_system = QA_System()
qa_pairs = qa_system.preprocess_and_create_qa_pairs(file_path)
question = input("Introduz a tua questão: ")
answer = qa_system.answer_question(question, file_path)
print(answer)
