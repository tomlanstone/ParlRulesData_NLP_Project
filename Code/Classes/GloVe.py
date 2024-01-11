import gc
from collections import defaultdict

import joblib
import numpy as np
from mittens import GloVe as GM
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class GloVe():
    def __init__(self, input_df):
        self.data = input_df.drop_duplicates(subset = "text_index")

    def preprocess(self, stop_words = False):
        if stop_words:
            self.documents_col = "text_preprocessed"
        else:
            self.documents_col = "text_no_stop"
            
        tqdm.pandas(desc = "Tokenising...")
        self.data.loc[:, "tokens"] = self.data[self.documents_col].progress_apply(word_tokenize) ## Will neeed to verify that this works, if not revert to with out the .loc
        return self.data

    def _co_matrix(self, window_size):
        co_occurrence_matrix = defaultdict(lambda: defaultdict(int)) ## Create an empty nested default dict
        print("Learning co-occurrences...")
        # Iterate over the tokens in the dataset
        for tokens in tqdm(self.data['tokens']):
            # Create a loop where i represents numbers from 0 to the total number of tokens
            for i in range(len(tokens)):
                # Determine the range of neighboring tokens based on the window size
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
                    # Skip the token if it is the same as the current token
                    if i != j:
                        # Retrieve the current tokens and update the co-occurrence count
                        word_i = tokens[i]
                        word_j = tokens[j]
                        dist = abs(len(tokens)-j)
                        co_occurrence_matrix[word_i][word_j] += 1/dist
        self.co_matrix = co_occurrence_matrix
        self.vocab = list(co_occurrence_matrix.keys())

    def _mat_to_array(self, vocab):
           # Create a NumPy array of zeros with dimensions based on the original matrix
        array = np.zeros((len(vocab), len(vocab)))
        # Iterate over the words and their indices
        pb = tqdm(total = len(vocab), desc = "Converting to Array...")
        for i, word_i in enumerate(vocab):
            for j, word_j in enumerate(vocab):
                # Assign the co-occurrence count from the matrix to the corresponding position in the array
                array[i][j] = self.co_matrix[word_i][word_j]
            pb.update(1)
        pb.close()
        self.co_matrix = array

    def make_co_matrix(self, window_size = 4):
        self._co_matrix(window_size)
        self._mat_to_array(self.vocab)

    def run_GloVe(self, dimensions = '100', iterations = 1500):
        self.dimensions = dimensions
        ## Train GloVe model
        model = GM(n = int(dimensions), max_iter = iterations, display_progress=500)
        print("Fitting GloVe model...")
        embeddings = model.fit(self.co_matrix)
        ## Convert to dictionary
        self.embeddings = {word: vector for word, vector in zip(self.vocab, embeddings)}
        del model
        gc.collect()

    def document_encode(self, documents = None):
        """
        Encode the provided documents or defaults to training data if none provided.
        """
        if not documents:
            documents = self.data
        # Determine the dimension of the embeddings
        embedding_dim = len(list(self.embeddings.items())[0][1])
        self.document_embeddings = [] # Empty list to hold embeddings
        for _, row in (documents.iterrows()):
            tokens = row["tokens"]
            # Get the GloVe embeddings for each token and handle missing tokens with zero-filled arrays
            embeddings = [self.embeddings.get(token, np.zeros(embedding_dim)) for token in tokens]
            # Calculate the average embedding for the document
            avg_embedding = np.mean(embeddings, axis=0)
            # Append the document embedding to the list
            self.document_embeddings.append(avg_embedding)

    def save_docs(self, directory, stop_words, name = None):
        stops = ""
        if not stop_words:
            stops = "_b"
        if not name:
            name = f"GloVe - {self.dimensions} dimensions{stops}.joblib"
        if not name.endswith(".joblib"):
            name = f"{name}.joblib"

        with open(f'{directory}/{name}', 'wb') as file:
            joblib.dump(self.document_embeddings, file)
            file.close()

    
 