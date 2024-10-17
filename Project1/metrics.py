import numpy as np
import pandas as pd
import ast

def calculate_mrr(relevant_docs, results):
    mrr_scores = []
    
    for query_id, retrieved_ids in results:  
        query_id_str = f'q{query_id}'  
        relevant_row = relevant_docs[relevant_docs[0] == query_id_str]
        
        if relevant_row.empty:
            print(f"Warning: No relevant documents found for query ID {query_id_str}")
            continue 
        relevant_ids = ast.literal_eval(relevant_row.iloc[0, 1])

        rank = next((j + 1 for j, doc_id in enumerate(retrieved_ids) if int(doc_id) in relevant_ids), None)
        
        if rank is not None:
            mrr_scores.append(1 / rank)  
        else:
            mrr_scores.append(0)  
    
    if len(mrr_scores) == 0:
        return 0, mrr_scores
    
    return np.mean(mrr_scores), mrr_scores

def calculate_map_and_recall(relevant_docs, results, k=10):
    map_scores = []
    recall_scores = []
    
    for query_id, retrieved_ids in results:
        query_id_str = f'q{query_id}'  
        
        relevant_row = relevant_docs[relevant_docs[0] == query_id_str]
        
        if relevant_row.empty:
            print(f"Warning: No relevant documents found for query ID {query_id_str}")
            continue  
        relevant_ids = ast.literal_eval(relevant_row.iloc[0, 1])

        num_relevant = 0  
        precisions = []
        
        for i, doc_id in enumerate(retrieved_ids[:k]):  
            if int(doc_id) in relevant_ids:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1)) 
        if precisions:
            ap = np.mean(precisions)  
        else:
            ap = 0
        
        map_scores.append(ap)  
        

        recall = num_relevant / len(relevant_ids) if len(relevant_ids) > 0 else 0
        recall_scores.append(recall)

    if len(map_scores) == 0:
        return 0, 0, [], [] 
    
    map_at_k = np.mean(map_scores)
    mean_recall_at_k = np.mean(recall_scores)
    
    return map_at_k, mean_recall_at_k, map_scores, recall_scores


