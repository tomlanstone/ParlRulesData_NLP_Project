from Classes.ConfusionMatrix import ConfusionMatrix
from Classes.Results import Results
import re
from tqdm import tqdm

if __name__ == "__main__":
   ## Locate the directory that data will be stored in, used for all loading and saving later on
    data_dir = "Data"  

    ## Define the path to the database of results
    db_path = f"{data_dir}/results.sqlite"
    results = Results(db_path)
    results.create_res_tab("kmeans_accuracies")
 
    pb = tqdm(total = 48, desc = "Evaluating Cluster Accuracy")
    for i in ["a","b","c","d"]:
        table = f"clusters_{i}"
        y_true = [int(r[0]) for r in results.get_col("root_num", table)]
        models = results.list_table_columns(table)    
        for m in models[3:]:
            m_name = re.search("^(.*?)(?=-)", m).group(0)
            m_type = re.search("^.*?-(.*)", m).group(1)

            y_pred = [int(r[0]) for r in results.get_col(f"`{m}`", table)]

            cm = ConfusionMatrix(y_true=y_true, y_pred=y_pred, name = f'{m}_{i}')
            acc = cm.accuracy()
           
            results.add_result("kmeans_accuracies",m_name,m_type,i,acc)
            pb.update(1)
