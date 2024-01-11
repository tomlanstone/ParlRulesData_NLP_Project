from nltk.corpus import stopwords
import nltk
import warnings
from time import time
from Classes.HCSOs import HCSOs
from Classes.Results import Results
from Classes.GloVe import GloVe
from Classes.Doc2Vec import Doc2Vec
from Classes.SBERT import SBERT
import pandas as pd
import os
from waiter import get_time

try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

if __name__ == "__main__":

    ## Locate the directory that data will be stored in, used for all loading and saving later on
    data_dir = "Data"  

    ## Define the path to the database of results
    db_path = f"{data_dir}/results.sqlite"
    results = Results(db_path)
    results.add_model_tab()

    raw_data = HCSOs(data_dir)
    raw_data.refine_data() ## Run the function to restructure the data to be more convenient for use

    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    times_df = pd.DataFrame()
    for b in [False, True]:
        if b:
            raw_data.preprocess()
            stops = "In"
            tag = ""
        else:
            raw_data.preprocess(stop_words=stop_words)
            stops = "Out"
            tag = "_b"
        
        print(f"Modelling GloVe with stop words {stops}")
        glove_model = GloVe(raw_data.processed)
        glove_model.preprocess(stop_words = b)
        glove_model.make_co_matrix()
        for i in ['025','050','075','100']:
            
            if os.path.exists(f"{data_dir}/Embeddings/GloVe - {i} dimensions{tag}.joblib"):
                print(f"Skipping GloVe - {i} dimensions{tag}")
                continue
            print(f"GloVe {i} dimesions")
            print(f"Starting at: {get_time()}")
            start = time()
            with warnings.catch_warnings(): ## Pandas hates append now, this just catches the warning
                warnings.simplefilter(action='ignore', category=UserWarning)
                glove_model.run_GloVe(dimensions=i, iterations = 3000)
            glove_model.document_encode()
            glove_model.save_docs(directory = f"{data_dir}/Embeddings", stop_words = b)
            row = {
                "model": "GloVe",
                "type": i,
                "stop_words": stops,
                "dimensions": int(i),
                "time": time() - start
            }
            results.add_training_time(row)
            print("\n")

        print(f"Modelling Doc2Vec with stop words {stops}")
        doc2vec_model = Doc2Vec(raw_data.processed)
        doc2vec_model.tag_docs(stop_words = b)
        for i in ['025','050','075','100']:

            if os.path.exists(f"{data_dir}/Embeddings/Doc2Vec - {i} dimensions{tag}.joblib"):
                print(f"Skipping Doc2Vec - {i} dimensions{tag}")
                continue
            print(f"Doc2Vec {i} dimesions")
            print(f"Starting at: {get_time()}")

            start = time()
            doc2vec_model.document_encode(dimensions=(i), iterations = 3000)
            doc2vec_model.save_docs(directory = f"{data_dir}/Embeddings", stop_words = b)
            row = {
                "model": "Doc2Vec",
                "type": i,
                "stop_words": stops,
                "dimensions": int(i),
                "time": time() - start
            }
            results.add_training_time(row)
            print("\n")
        
        print(f"Modelling SBERT with stop words {stops}")
        SBERT_model = SBERT(raw_data.processed)
        for i in ['bert-base-nli-mean-tokens','all-mpnet-base-v2','all-MiniLM-L6-v2','gtr-t5-xxl']:

            if os.path.exists(f"{data_dir}/Embeddings/SBERT - {i}{tag}.joblib"):
                print(f"Skipping SBERT - {i}{tag}")
                continue
            print(f"SBERT {i}")            
            print(f"Starting at: {get_time()}")

            start = time()
            SBERT_model.document_encode(model = i, stop_words = b)
            SBERT_model.save_docs(directory = f"{data_dir}/Embeddings", tag = tag)
            row = {
                "model": "SBERT",
                "type": i,
                "stop_words": stops,
                "dimensions": SBERT_model.dimensions,
                "time": time() - start
            }
            results.add_training_time(row)
            print("\n")