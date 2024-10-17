import os  
import nltk 
import jieba  
from nltk.stem.porter import PorterStemmer
from gensim.utils import simple_preprocess 
from nltk.stem import SnowballStemmer  
import numpy as np 

from tfidfcalculation import *

def preprocess(text,stopwords):
    stemmer = PorterStemmer()
    tokens = simple_preprocess(text, deacc=True) 
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
    return tokens

def preprocess_cn(text,stopwords):
    tokens = jieba.lcut(text)
    
    processed_tokens = [token for token in tokens if token not in stopwords]
    
    return processed_tokens
def load_documents(folder,stopwords):
    documents = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                documents[file] = preprocess(f.read(),stopwords)
    return documents
    
def load_documents_cn(folder,stopwords):
    documents = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                documents[file] = preprocess_cn(f.read(),stopwords)
    return documents
def load_queries(folder,stopwords=None):
    queries = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                queries.append(preprocess(f.read(),stopwords=stopwords)) 
    return queries
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())  
    return stopwords
    
def build_vocabulary(documents):
    vocabulary = set()
    for doc in documents.values():
        vocabulary.update(doc)
    return list(vocabulary)

def extract_nouns_verbs(tokens):
    tagged = nltk.pos_tag(tokens)
    return [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('VB')]

def adjust_query_vector(original_query, feedback_terms,stopwords=None,term_index=None,idf_vector=None):
    original_tokens = preprocess(original_query,stopwords=stopwords)
    feedback_tokens = preprocess(" ".join(feedback_terms),stopwords=stopwords)
    
    original_tf = calculate_tf(original_tokens, term_index)
    feedback_tf = calculate_tf(feedback_tokens, term_index)
    
    adjusted_query_vector = 1 * original_tf + 0.5 * feedback_tf
    adjusted_query_tfidf = adjusted_query_vector * idf_vector
    return adjusted_query_tfidf