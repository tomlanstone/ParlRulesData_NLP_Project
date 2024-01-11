from gensim.models.doc2vec import Doc2Vec as D2V, TaggedDocument
import joblib
import gc

class Doc2Vec():
    def __init__(self, input_df):
        self.data = input_df.drop_duplicates(subset = "text_index")
    
    def tag_docs(self, stop_words = False):
        if  stop_words:
            self.documents_col = "text_preprocessed"
        else:
            self.documents_col = "text_no_stop"
        self.tagged_docs = [TaggedDocument(words=doc.split(), tags=[i]) for i, doc in enumerate(self.data[self.documents_col])]

    def document_encode(self, dimensions = 100, iterations = 1500):
        self.dimensions = dimensions
        
        model = D2V(vector_size = int(dimensions), window = 4, epochs = iterations, dm = 0) ## DM = 0 for DBOW
        model.build_vocab(self.tagged_docs)
        model.train(self.tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        self.document_embeddings = [model.infer_vector(doc.split()) for doc in self.data[self.documents_col]]
        del model
        gc.collect()
    
    def save_docs(self, directory, stop_words, name = None):
        stops = ""
        if not stop_words:
            stops = "_b"
        if not name:
            name = f"Doc2Vec - {self.dimensions} dimensions{stops}.joblib"
        if not name.endswith(".joblib"):
            name = f"{name}.joblib"

        with open(f'{directory}/{name}', 'wb') as file:
            joblib.dump(self.document_embeddings, file)
            file.close()