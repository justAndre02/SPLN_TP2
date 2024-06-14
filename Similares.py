from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import psycopg2
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# Baixar as stop words da NLTK
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

db_config = {
    'dbname': 'diariorepublica',
    'user': 'postgres',
    'password': '1597535',
    'host': 'localhost',
    'port': '5432'
}

# Conectar ao banco de dados PostgreSQL
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Carregar resumos dos documentos
cursor.execute("SELECT id, notes FROM dreapp_document")
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['id', 'notes'])

# Vectorizar textos
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(df['notes'])

# Função de similaridade
def search(query, tfidf_matrix, vectorizer, df):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = similarities.argsort()[::-1]
    return df.iloc[related_docs_indices]

# Exemplo de busca
query = "Tribunal Constitucional"
result = search(query, tfidf_matrix, vectorizer, df)
print(result.head(10))

# Função para obter texto completo pelo document_id
def get_full_text(document_id):
    cursor.execute("SELECT html_text FROM dreapp_documenttext WHERE document_id=%s", (document_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# Exemplo de obtenção de textos completos
document_ids = result['id'][:10]  # Supondo que 'result' é o DataFrame do passo anterior
texts = [get_full_text(doc_id) for doc_id in document_ids]

# Carregar modelo de Q&A
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Função de Q&A
def answer_question(question, context):
    response = qa_model(question=question, context=context)
    return response['answer']

# Exemplo de uso
question = "Quem foi eleito para o Tribunal Constitucional?"
answers = [answer_question(question, text) for text in texts]
print(answers)


