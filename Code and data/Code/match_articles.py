from Classes.HCSOs import HCSOs
from Classes.Embeddings import Embeddings
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import numpy as np
import pandas as pd

def match_articles(model = "SBERT - all-mpnet-base-v2.joblib"):

    article_vectors = Embeddings(f"Data/Embeddings/{model}",model)
    article_vectors.load()
    article_vectors.scale()

    hcso = HCSOs("Data", "parlrules_ukhoc_3.0.1_articles.csv")
    hcso.refine_data()
    hcso.add_version_id()

    results_data = []
    pb = tqdm(total = 253310, desc = "Matching Documents") ## haven't done the maths to get the total yet, just run it without one the first time and copy in the final number of iterations for subsequent tries
    for _i , row_i in hcso.unique_text_ids(full = True).iterrows():

        vec_i = article_vectors.data[row_i["text_index"]].reshape(1, -1)

        match_sim_val  = np.inf  
        match_sim_text_index = None
        match_sim_root = None

        version = hcso.data.query(f'version_num == {row_i["version_num"] - 1}')

        for _ii, row_ii in version.iterrows():

            vec_ii = article_vectors.data[row_ii["text_index"]].reshape(1, -1)

            e = euclidean_distances(vec_i, vec_ii)[0][0]

            if e < match_sim_val:
                    match_sim_val  = e
                    match_sim_text_index = row_ii["text_index"]
                    match_sim_root = row_ii["root_num"]
            pb.update(1)
            
        results_data.append([row_i["text_index"], row_i["root_num"], match_sim_text_index, match_sim_root, match_sim_val])

    results = pd.DataFrame(results_data, columns=['text_index', 'text_root', 'match_index', 'match_root', 'match_sim'])
    pb.close()
    
    return results

def match_rate(results):
    return (results['text_root'] == results['match_root']).mean() * 100

if __name__ == "__main__":
     m = match_articles()
     print(match_rate(m))
