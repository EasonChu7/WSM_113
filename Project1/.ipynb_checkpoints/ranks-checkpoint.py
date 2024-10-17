import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances  
from sklearn.preprocessing import normalize
from tools import *
from tfidfcalculation import *

def calculate_euclidean_distance(query_vector, doc_vector):
    """手算 Euclidean distance"""
    return np.sqrt(np.sum((query_vector - doc_vector) ** 2))
    
def rank_documents(query, doc_vectors, method='cosine', vector_type='tfidf', top_n=10, stopwords=None, term_index=None, idf_vector=None):
    query_tokens = preprocess(query, stopwords)
    
    if vector_type == 'tfidf':
        query_tf = calculate_tf(query_tokens, term_index)
        query_vector = calculate_tfidf(query_tf, idf_vector)
    else:
        query_vector = calculate_tf(query_tokens, term_index)

    query_vector = query_vector.reshape(1, -1)
    
    scores = []
    for doc_id, doc_vector in doc_vectors.items():
        doc_vector = doc_vector.reshape(1, -1)
        
        if method == 'euclidean':
            dist = calculate_euclidean_distance(query_vector, doc_vector)
            scores.append((doc_id, dist))
        elif method == 'cosine':
            sim = cosine_similarity(query_vector, doc_vector)[0][0]
            scores.append((doc_id, sim))

    if method == 'cosine':
        ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)  
    elif method == 'euclidean':
        ranked_docs = sorted(scores, key=lambda x: x[1]) 
    
    return ranked_docs[:top_n]

def rank_all_methods(query, tf_vectors, tfidf_vectors, top_n=10,stopwords=None,term_index=None,idf_vector=None):
    print(f"Ranking results for query: '{query}'")
    
    # 1. TF + Euclidean
    ranked_tf_euclidean = rank_documents(query, tf_vectors, method='euclidean', vector_type='tf', top_n=top_n,stopwords=stopwords,term_index=term_index)
    print("\n1. TF + Euclidean Ranking (Top 10):")
    for doc_id, score in ranked_tf_euclidean:
        print(f"{doc_id}: {score:.4f}")
    
    # 2. TF-IDF + Euclidean
    ranked_tfidf_euclidean = rank_documents(query, tfidf_vectors, method='euclidean', vector_type='tfidf', top_n=top_n,stopwords=stopwords,term_index=term_index,idf_vector=idf_vector)
    print("\n2. TF-IDF + Euclidean Ranking (Top 10):")
    for doc_id, score in ranked_tfidf_euclidean:
        print(f"{doc_id}: {score:.4f}")
    
    # 3. TF + Cosine Similarity
    ranked_tf_cosine = rank_documents(query, tf_vectors, method='cosine', vector_type='tf', top_n=top_n,stopwords=stopwords,term_index=term_index)
    print("\n3. TF + Cosine Similarity Ranking (Top 10):")
    for doc_id, score in ranked_tf_cosine:
        print(f"{doc_id}: {score:.4f}")
    
    # 4. TF-IDF + Cosine Similarity
    ranked_tfidf_cosine = rank_documents(query, tfidf_vectors, method='cosine', vector_type='tfidf', top_n=top_n,stopwords=stopwords,term_index=term_index,idf_vector=idf_vector)
    print("\n4. TF-IDF + Cosine Similarity Ranking (Top 10):")
    for doc_id, score in ranked_tfidf_cosine:
        print(f"{doc_id}: {score:.4f}")
    
    return ranked_tfidf_cosine  



def rank_documents_with_adjusted_query(adjusted_query_vector, doc_vectors, top_n=10):
    query_vector = adjusted_query_vector.reshape(1, -1)
    
    similarities = []
    for doc_id, doc_vector in doc_vectors.items():
        doc_vector = doc_vector.reshape(1, -1)
        sim = cosine_similarity(query_vector, doc_vector)[0][0]
        similarities.append((doc_id, sim))
    
    ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return ranked_docs