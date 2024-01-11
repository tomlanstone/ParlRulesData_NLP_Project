from Classes.Results import Results
from Classes.HCSOs import HCSOs
from Classes.Embeddings import Embeddings
import os
from sklearn.metrics.pairwise import euclidean_distances
from waiter import get_time
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sim_by_group_full(grouping, embeddings, articles):
    results = []

    for v in articles[grouping].unique():
        version = articles.query(f"{grouping} == {v}")

        version = version.drop_duplicates(subset = "text_index")
        
        similarities = []

        for _i, article_i in enumerate(version.to_numpy()):
            text_index_i = article_i[9]
            vec_i = embeddings[text_index_i].reshape(1, -1)

            for _ii, article_ii in enumerate(version.to_numpy()):
                if _ii <= _i:  # ensure that _ii is always greater than _i, so we don't compare the same pairs twice
                    continue
                text_index_ii = article_ii[9]
                vec_ii = embeddings[text_index_ii].reshape(1, -1)

                e = euclidean_distances(vec_i, vec_ii)[0][0]
                similarities.append(e)

        # Storing all values for each group
        group_result = {
            grouping: v,
            "similarity": similarities
        }
        results.append(group_result)

    return results

def find_assumption_failures(root_results, version_results):
    failures = []
    total = 0
    # Determine the least similar cosine
    for root in root_results:
        if len(root["similarity"]) == 0:
            continue
        least_similar = max(root["similarity"])

        for version in version_results:
            if min(version["similarity"]) < least_similar:
                failure = {
                    "root": root["root_num"],
                    "version": version["version_num"]
                }
                failures.append(failure)
            total += 1
    return failures, total

def prepare_data_for_boxplot(results, category_name):
    similarities = []
    categories = []

    for result in results:
        similarities.extend(result["similarity"])
        categories.extend([category_name] * len(result["similarity"]))
    
    return pd.DataFrame({
        "Category": categories,
        "similarity": similarities
    })

if __name__ == "__main__":
    
    ## Locate the directory that data will be stored in, used for all loading and saving later on
    data_dir = "Data"  
    ## Define the path to the database of results
    db_path = f"{data_dir}/results.sqlite"
    ## List out the files in the embedding folder
    embedding_files = [i for i in os.listdir(f"{data_dir}/Embeddings")]

    ## Initiate the Results class object that points to and interacts with the results database
    results = Results(db_path)
    results.drop_tab("kmeans_times")
    results.create_res_tab("kmeans_times")
    ## Initiate the HCSOs object that holds and manipulates the Standing Orders original data file
    raw_data = HCSOs(data_dir, "parlrules_ukhoc_3.0.1_articles.csv")
    raw_data.refine_data()
    raw_data.add_version_id()
    results_dict = {}
    for i in embedding_files:
        
        ## Ignore filees that don't have the joblib extension used to save the embeddings
        if not i.endswith('.joblib'):
            continue
        m = i.replace(".joblib","")
        print(m)
        print(get_time())
        ## Initial set up for the embeddings
        emb_path = f"{data_dir}/Embeddings/{i}" ## Place the embedding file is located
        embeddings = Embeddings(emb_path, i) ## Initiate document, the Embeddings object holding the embeddings
        embeddings.load() ## Loads the embeddings. The embeddings are a list of numpy arrays
        embeddings.scale() ## Scales the loaded embeddings 

        if not os.path.exists(f"{data_dir}/sims/{m} versions.joblib"):
            version_results = sim_by_group_full("version_num", embeddings.data, raw_data.data)
            with open(f"{data_dir}/sims/{m} versions.joblib", "wb") as f:
                joblib.dump(version_results,f)
        else: 
            with open(f"{data_dir}/sims/{m} versions.joblib", "rb") as f:
                version_results = joblib.load(f)
        if not os.path.exists(f"{data_dir}/sims/{m} roots.joblib"):
            root_results = sim_by_group_full("root_num", embeddings.data, raw_data.data)
            with open(f"{data_dir}/sims/{m} roots.joblib", "wb") as f:
                joblib.dump(root_results,f)
        else:
            with open(f"{data_dir}/sims/{m} roots.joblib", "rb") as f:
                root_results = joblib.load(f)

        failures, total = find_assumption_failures(root_results, version_results)
       
        results_dict[m] = len(failures)/total

        root_df = prepare_data_for_boxplot(root_results, "Root")
        version_df = prepare_data_for_boxplot(version_results, "Version")
        df = pd.concat([root_df, version_df])

        sns.set_context("talk")
        # Boxplot for Euclidean Distance
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x= "Category", y="Euclidean Distance", data=df)
        ax.set_xlabel("")
        #plt.title("Distribution of Euclidean Distance for Roots vs. Versions")
        plt.savefig(f"{data_dir}/Graphs/Euclidean Distance {m}.png")
        plt.close()
    results_df = pd.DataFrame(results_dict, index=results_dict.keys())
    results_df.to_csv(f"{data_dir}/sims.csv")