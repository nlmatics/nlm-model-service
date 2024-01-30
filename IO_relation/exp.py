from IO_process import passages_mat, queries_mat, passages_mat_no_count, queries_mat_no_count, queries
from utils import query_Score, score_qp, score_qp_vol, T1, T2, T3, Retrieval_result
import pandas as pd
import numpy as np



top_k = [1, 5, 10, 20, 50, 100]

Methods = [
    "Exact Match", 
    ".7<",
    "soft thresholding, alpha = 5",
    "soft thresholding, alpha = 10",
    "soft thresholding, alpha = 20"
    
]


scoring_method = [
    lambda x,y: score_qp(x, y, T1),
    lambda x,y: score_qp(x, y, T2),
    lambda x,y: score_qp(x, y, lambda z: T3(z, alpha = 5)),
    lambda x,y: score_qp(x, y, T3),
    lambda x,y: score_qp(x, y, lambda z: T3(z, alpha = 20))

   
]

R_with_multiplicity = []
R_without_multiplicity = []


idx = np.random.choice( np.arange(len(queries_mat)), 500, replace = False)


for s, score_func in enumerate(scoring_method):
    
    R_with_multiplicity.append([])
    R_without_multiplicity.append([])
    
    print("=" * 10 + Methods[s] + "="*10) 
    for i , q in enumerate( map(queries_mat.__getitem__, idx) ):
        if i % 50 == 1:
            print(f"{i} out of {len( idx )}")
        
        R_with_multiplicity[-1].append(
            [ query_Score(q, passages_mat, score_func) ]
        )

        R_without_multiplicity[-1].append(
            [ query_Score(q, passages_mat_no_count, score_func) ]
        )

                                        
idx_with_multiplicity = np.argsort(- np.array(R_with_multiplicity), axis = -1)
idx_without_multiplicity = np.argsort(- np.array(R_without_multiplicity), axis = -1)



Labels = np.array(list(queries.values()))[ idx ]



Restult_with_multiplicity = Retrieval_result(idx_with_multiplicity, Labels)
Restult_without_multiplicity = Retrieval_result(idx_without_multiplicity, Labels)





df_with_multiplicity = pd.DataFrame(
    data=np.array(Restult_with_multiplicity).T,
    index = Methods, 
    columns = top_k
    )


df_without_multiplicity = pd.DataFrame(
    data=np.array(Restult_without_multiplicity).T,
    index = Methods, 
    columns = top_k
)



df_with_multiplicity.to_csv("./df_with_multiplicity_train_set")

df_without_multiplicity.to_csv("./df_without_multiplicity_train_set")