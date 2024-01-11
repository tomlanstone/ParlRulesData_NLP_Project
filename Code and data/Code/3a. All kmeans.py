'''
This script runs all four kmeans experiments for each set of embeddings and saves the results to the results database.

It begins by defining all relevant classes and functions, scroll to the bottom for the main loop.
'''
import os
import warnings
from time import time

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from Classes.Embeddings import Embeddings
from Classes.HCSOs import HCSOs
from Classes.Results import Results



def expand_embeddings(input_embeddings, input_df):
    '''
    Used to take the embeddings output by the models for each unique text and expand them to match the order of texts in the full data set

    embeddings is the list of numpy arrays holding the document embeddings

    input_df is 
    '''
    df = input_df.copy()
    embeddings_df = pd.DataFrame(input_embeddings)
    embeddings_df["text_index"] = [index for index, _ in enumerate(embeddings_df.to_numpy())]
    merged_df = df.merge(embeddings_df, on='text_index', how='left')
    output_embeddings = [row.to_numpy() for _, row in merged_df.iloc[:,input_df.shape[1]:].iterrows()]
    assert(len(output_embeddings[0]) == len(input_embeddings[0]))
    return output_embeddings

def time_kmeans(results_dict, kmeans_obj, document, embeddings):
    with warnings.catch_warnings(): ## Ignore the warning about memory leaks. It doesn't cause a problem on my machine but for anyone else running it, get rid of the warning filter to see the warning if it starts causing a memory leak
        start = time()
        warnings.simplefilter(action='ignore', category=UserWarning) 
        kmeans_obj.fit(embeddings)
    results_dict[f"{document.model}-{document.type}"] = kmeans_obj.labels_
    return time() - start

if __name__ == "__main__":

    ## Locate the directory that data will be stored in, used for all loading and saving later on
    data_dir = "Data"  
    ## Define the path to the database of results
    db_path = f"{data_dir}/results.sqlite"
    ## List out the files in the embedding folder
    embedding_files = [i for i in os.listdir(f"{data_dir}/Embeddings")]

    ## Initiate the Results class object that points to and interacts with the results database
    results = Results(db_path)
    #results.drop_tab("kmeans_times")
    results.create_res_tab("kmeans_times")
    ## Initiate the HCSOs object that holds and manipulates the Standing Orders original data file
    raw_data = HCSOs(data_dir, "parlrules_ukhoc_3.0.1_articles.csv")
    raw_data.refine_data() ## Run the function to restructure the data to be more convenient for use

    ## Initiate a kmeans object that will be used to fit every kmeans model
    kmeans = KMeans(n_clusters = 323, n_init = 25, random_state = 42) 

    ## Get the two different versions of the text index column
    id_unique =  raw_data.unique_text_ids().copy().reset_index().drop("index", axis = 1)
    id_all = raw_data.all_ids().copy()

    ## Create dictionaries that will hold the results of each different kmeans configuration
    results_dict_a = {} ## All articles, no stop word filter
    results_dict_b = {} ## Unique articles, no stop word filter
    results_dict_c = {} ## All articles, stop words filtered out
    results_dict_d = {} ## Unique articles, stop words filtered out
    ## Create a dataframe that run times will be held in
    #times_df = pd.DataFrame(columns = ["model","type","a","b","c","d"])
    ## Initiate a progress bar (4 experiments x 12 embedding models = 48 total loops)
    pb = tqdm(total=48)

    ## Main K means loop, iterates over the contents of the embedding folder 
    for i in embedding_files:
        
        ## Ignore filees that don't have the joblib extension used to save the embeddings
        if not i.endswith('.joblib'):
            continue
        
        ## Initial set up for the embeddings
        emb_path = f"{data_dir}/Embeddings/{i}" ## Place the embedding file is located
        embeddings = Embeddings(emb_path, i) ## Initiate document, the Embeddings object holding the embeddings
        embeddings.load() ## Loads the embeddings. The embeddings are a list of numpy arrays
        embeddings.scale() ## Scales the loaded embeddings 

        ## Define the two versions of the data to be used
        data_short = embeddings.data ## Just unique embeddings
        data_long = expand_embeddings(embeddings.data, raw_data.all_ids()) ## Embeddings expanded such that each of thee 37947 articles is represented 
        ## Checks to see if the model being used was trained with or without stop words. If stop words were not removed, run the experiments with labels a and b, otherwise c and d.
        if not embeddings.no_stop_words:

            ## experiment_a
            ktime = time_kmeans(results_dict_a, kmeans, embeddings, data_long)
            try:
                results.add_result("kmeans_times", embeddings.model, embeddings.type, "a", ktime)
            except Exception as e:
                print(e)
            pb.update(1)

            ## experiment_b
            ktime = time_kmeans(results_dict_b, kmeans, embeddings, data_short)
            try:
                results.add_result("kmeans_times", embeddings.model, embeddings.type, "b", ktime)
            except Exception as e:
                print(e)
            pb.update(1)

        else:

            ## experiment_c
            ktime = time_kmeans(results_dict_c, kmeans, embeddings, data_long)
            try:
                results.add_result("kmeans_times", embeddings.model, embeddings.type, "c", ktime)
            except Exception as e:
                print(e)
            pb.update(1)

            ## experiment_d
            ktime = time_kmeans(results_dict_d, kmeans, embeddings, data_short)
            try:
                results.add_result("kmeans_times", embeddings.model, embeddings.type, "d", ktime)
            except Exception as e:
                print(e)
            pb.update(1)

    ## Close the progress bar
    pb.close()

    a = pd.DataFrame(results_dict_a)
    b = pd.DataFrame(results_dict_b)
    c = pd.DataFrame(results_dict_c)
    d = pd.DataFrame(results_dict_d)
    if False:
    ## Save all the results to the db. The tables are created by turning the dictionaries into dataframes with the text indexes and root numbers attached at the start.
        results.overwrite_table(pd.concat([id_all,a], axis = 1),"clusters_a")
        results.overwrite_table(pd.concat([id_unique,b], axis = 1),"clusters_b")
        results.overwrite_table(pd.concat([id_all,c], axis = 1),"clusters_c")
        results.overwrite_table(pd.concat([id_unique,d], axis = 1),"clusters_d")
    