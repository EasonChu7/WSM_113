import numpy as np  
from collections import Counter  
import math 

def calculate_tf_weighting(doc, term_index):
    term_counts = Counter(doc)  
    total_terms = len(doc) 
    max_freq = max(term_counts.values()) if term_counts else 1  
    
    tf_vector = np.zeros(len(term_index))
    
    for term, count in term_counts.items():
        if term in term_index:  
            idx = term_index[term]
            tf_vector[idx] = 0.5 + 0.5 * (count / max_freq)  # Refer to the course slide 0.5+0.5*tf/maxFreq
    
    return tf_vector

def calculate_tf(doc, term_index):
    tf_vector = np.zeros(len(term_index))
    term_counts = Counter(doc)
    total_terms = len(doc)
    for term, count in term_counts.items():
        if term in term_index: 
            idx = term_index[term]
            tf_vector[idx] = count / total_terms
    return tf_vector

def calculate_idf(documents, term_index):
    N = len(documents)
    idf_vector = np.zeros(len(term_index))
    for term, idx in term_index.items():
        df = sum(1 for doc in documents.values() if term in doc)
        idf_vector[idx] = math.log((N + 1) / (df + 1)) + 1  
    return idf_vector

def calculate_tfidf(tf_vector, idf_vector):
    return tf_vector * idf_vector

