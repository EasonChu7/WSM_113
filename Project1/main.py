import argparse
import nltk
from gensim.parsing.preprocessing import STOPWORDS
from tools import *
from ranks import *
from tfidfcalculation import *
from metrics import *
from tqdm import tqdm
import time
def main():
    
    parser = argparse.ArgumentParser(description="Process English and Chinese queries.")
    parser.add_argument('--Eng_query', type=str, required=False, help='English query string')
    parser.add_argument('--Chi_query', type=str, required=False, help='Chinese query string')
    
    args = parser.parse_args()

    
    eng_query = args.Eng_query
    ch_query = args.Chi_query

    #nltk.download('all')
    
    print(f"English Query: {eng_query}")
    print(f"Chinese Query: {ch_query}")
    if eng_query:
        print("processing English News")
        with open('EnglishStopwords.txt', 'r') as f:
            custom_stopwords = set(f.read().splitlines())
        
        stopwords = STOPWORDS.union(custom_stopwords)
        

        documents = load_documents('./EnglishNews',stopwords=stopwords)
        vocab = build_vocabulary(documents)
        print('vocabulary setup finished!')
        print('Constructing term index!')
        term_index = {term: idx for idx, term in enumerate(vocab)}
        tf_vectors = {}
        tf_vectors_normalized = {}
        
        print('Constructing TF vectors!')
        
        for doc_id, doc in documents.items():
            tf_vectors[doc_id] = calculate_tf(doc, term_index)
            tf_vectors_normalized[doc_id] = calculate_tf_weighting(doc, term_index)
        print('Constructing TF-IDF vectors!')
        idf_vector = calculate_idf(documents, term_index)
        
        tfidf_vectors = {}
        for doc_id, tf_vector in tf_vectors.items():
            tfidf_vectors[doc_id] = calculate_tfidf(tf_vector, idf_vector)
        print('Start Ranking process!')
        ranked_tfidf_cosine = rank_all_methods(eng_query, tf_vectors_normalized, tfidf_vectors, top_n=10,stopwords=stopwords,term_index=term_index,idf_vector=idf_vector)
        
        print("\nQ2: Relevance feedback with Cosine Similarity + TF-IDF")

        top_doc_id = ranked_tfidf_cosine[0][0]
        top_doc_tokens = documents[top_doc_id]
        feedback_terms = extract_nouns_verbs(top_doc_tokens)
        
        adjusted_query_tfidf = adjust_query_vector(eng_query, feedback_terms,stopwords=stopwords,term_index=term_index,idf_vector=idf_vector)

        ranked_docs_with_feedback = rank_documents_with_adjusted_query(adjusted_query_tfidf, tfidf_vectors, top_n=10)
        
        print("Ranked documents after pseudo-feedback (Top 10):")
        for doc_id, score in ranked_docs_with_feedback:
            print(f"{doc_id}: {score:.4f}")
    if ch_query:
        print("processing Chinese News")
        stopwords_cn = load_stopwords("ChineseStopwords.txt")
        documents = load_documents_cn('./ChineseNews',stopwords=stopwords_cn )
        vocab = build_vocabulary(documents)
        term_index = {term: idx for idx, term in enumerate(vocab)}
        tf_vectors = {}
        tf_vectors_normalized = {}
        
        for doc_id, doc in documents.items():
            tf_vectors[doc_id] = calculate_tf(doc, term_index)
            tf_vectors_normalized[doc_id] = calculate_tf_weighting(doc, term_index)
        
        idf_vector = calculate_idf(documents, term_index)
        
        tfidf_vectors = {}
        for doc_id, tf_vector in tf_vectors.items():
            tfidf_vectors[doc_id] = calculate_tfidf(tf_vector, idf_vector)
        ranked_tfidf_cosine = rank_all_methods(ch_query, tf_vectors_normalized, tfidf_vectors, top_n=10,stopwords=stopwords_cn,term_index=term_index,idf_vector=idf_vector)
        
    print("Q4 Start")
    with open('EnglishStopwords.txt', 'r') as f:
        custom_stopwords = set(f.read().splitlines())
    stopwords = STOPWORDS.union(custom_stopwords)
    relevant_docs = pd.read_csv('./smaller_dataset/rel.tsv', sep='\t', header=None)
    documents = load_documents_q4('./smaller_dataset/collections',stopwords=stopwords)
    queries = load_queries('./smaller_dataset/queries',stopwords=stopwords)
    vocab = build_vocabulary(documents)
    term_index = {term: idx for idx, term in enumerate(vocab)}
    tf_vectors = {}

    for doc_id, doc in documents.items():
        tf_vectors[doc_id] = calculate_tf(doc, term_index)
    
    idf_vector = calculate_idf(documents, term_index)
    
    tfidf_vectors = {}
    for doc_id, tf_vector in tf_vectors.items():
        tfidf_vectors[doc_id] = calculate_tfidf(tf_vector, idf_vector)

    results = []

    for query_id, query_tokens in tqdm(queries):  
        query_tf = calculate_tf(query_tokens, term_index)  
        query_tfidf = calculate_tfidf(query_tf, idf_vector)  
        similarities = cosine_similarity([query_tfidf], list(tfidf_vectors.values())).flatten()

        top_10_indices = np.argsort(similarities)[::-1][:10].astype(int)

        top_10_doc_ids = [list(tfidf_vectors.keys())[idx] for idx in top_10_indices]

        results.append((query_id, top_10_doc_ids))  

    mrr_score, mrr_scores = calculate_mrr(relevant_docs, results)
    map_at_10, recall_at_10, map_scores, recall_scores = calculate_map_and_recall(relevant_docs, results)

    print(f"MRR@10: {mrr_score:.4f}")
    print(f"MAP@10: {map_at_10:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    
    
    
        
        
        


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    cost = end_time - start_time
    print(f'All execution cost {cost}s')
