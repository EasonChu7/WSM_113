import numpy as np
import pandas as pd

def calculate_mrr(relevant_docs, results):
    mrr_scores = []
    
    for i in range(len(relevant_docs)):
        relevant_ids = eval(relevant_docs[1][i])  
        retrieved_ids = results[i]  
        rank = next((j + 1 for j, doc_id in enumerate(retrieved_ids) if doc_id in relevant_ids), None)
        
        if rank is not None:
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0) 
    
    return np.mean(mrr_scores)

def calculate_map(relevant_docs, results):
    map_scores = []
    
    for i in range(len(relevant_docs)):
        relevant_ids = eval(relevant_docs[1][i]) 
        retrieved_ids = results[i]  
        hits = 0
        precisions = []
        
        for j, doc_id in enumerate(retrieved_ids[:10]):  
            if doc_id in relevant_ids:
                hits += 1
                precisions.append(hits / (j + 1))
        
        if hits > 0:
            average_precision = np.mean(precisions)
        else:
            average_precision = 0
        
        map_scores.append(average_precision)
    
    return np.mean(map_scores)

def calculate_recall(relevant_docs, results):
    recall_scores = []
    
    for i in range(len(relevant_docs)):
        relevant_ids = eval(relevant_docs[1][i])  
        retrieved_ids = results[i]  
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids[:10])  
        true_positives = len(relevant_set.intersection(retrieved_set))
        recall = true_positives / len(relevant_set) if relevant_set else 0
        recall_scores.append(recall)
    
    return np.mean(recall_scores)