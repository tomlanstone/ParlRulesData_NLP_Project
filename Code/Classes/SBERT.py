from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import joblib 
import gc

class SBERT():
    def __init__(self, input_df):
        self.data = input_df.drop_duplicates(subset = "text_index")
        self.pb_length = self.data.shape[0]

    def document_encode(self, model, stop_words):
        self.model = model
        if stop_words:
            self.documents_col = "text_preprocessed"
        else:
            self.documents_col = "text_no_stop"
        model = SentenceTransformer(self.model)
        document_embeddings = []
        progress_bar = tqdm(total=self.pb_length, desc = "Encoding Documents")
        for text in self.data[self.documents_col]:
            # Encode each document
            encoded_text = model.encode([text])[0]
            document_embeddings.append(encoded_text)

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        self.document_embeddings = [i for i in np.array(document_embeddings)]
        try:
            self.dimensions = len(self.document_embeddings[0])
        except: ## Seems to be working fine, get rid of this bit later
            print("Dimension calculator broke again")
            self.dimensions = 768
            if self.model == 'all-MiniLM-L6-v2':
                self.dimensions = 384
        
        ## Not entirely sure how the looping handles the most recently used model so this makes sure it gets deleted after use
        del model
        gc.collect()

    def save_docs(self, directory, tag,  name = None):
        """
        saves the document embeddings, fairly self explanatory
        """

        if not name:
            name = f"SBERT - {self.model}{tag}.joblib"
        if not name.endswith(".joblib"):
            name = f"{name}.joblib"

        with open(f'{directory}/{name}', 'wb') as file:
            joblib.dump(self.document_embeddings, file)
            file.close()