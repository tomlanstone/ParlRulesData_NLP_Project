from match_articles import match_articles, match_rate
import os
import pandas as pd

if __name__ == "__main__":
    results = []
    for i in os.listdir(f"data/Embeddings"):
        if not i.endswith(".joblib"):
            continue
        mod = i.replace(".joblib","")
        if mod.endswith("_b"):
            stopwords = "Out"
            mod = mod.replace("_b","")
        else:
            stopwords = "In"
        

        r = match_articles(i)
        mr = match_rate(r)
        results.append([mod,stopwords,mr])

    df = pd.DataFrame(results, columns=['Model', "Stop Words", 'Accuracy']).sort_values(by='Accuracy').reset_index(drop=True)
    df.to_latex("Data/another table.tex")
    
