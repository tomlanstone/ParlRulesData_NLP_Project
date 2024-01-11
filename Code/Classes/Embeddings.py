from sklearn.preprocessing import StandardScaler
import re
import joblib

class Embeddings():
    def __init__(self, path, name):
        self.path = path
        self._clean_name(name)

    def _clean_name(self,name):
        n = re.search(r'[^/]+$', self.path).group(0)
        n = name.replace(".joblib","")
        if n[-2:] == "_b":
            n = n[0:-2]
            self.no_stop_words = True
        else:
            self.no_stop_words = False
        n = n.replace(" ", "").replace("dimensions", "")
        self.model, self.type = n.split("-", 1)
        
    def load(self):
        if hasattr(self, "data"):
            return
        with open(self.path, "rb") as file:
            try:
                embeddings = joblib.load(file)
            except Exception as e:
                embeddings = False
                print(f"Loading failure. error message: {e}\nFile path: {self.path}")
            file.close()

        self.data = embeddings
    
    def scale(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)