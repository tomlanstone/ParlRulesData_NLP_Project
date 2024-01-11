from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np

class ConfusionMatrix:
    def __init__(self, y_true, y_pred, name):
        self.y_true = y_true
        self.y_pred = y_pred
        original_cm = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-original_cm)
        # Using the outputs of linear_sum_assignment
        self.adjusted_y_pred = np.empty_like(self.y_pred)
        for cluster_idx, true_label_idx in enumerate(col_ind):
            self.adjusted_y_pred[self.y_pred == cluster_idx] = true_label_idx

        self.cm = original_cm[row_ind[:, np.newaxis], col_ind]
        self.name = name
    
    def accuracy(self):
        # Calculate accuracy
        self.acc = np.diagonal(self.cm).sum() / self.cm.sum()
        output = round(self.acc * 100, 2)
        return output

    def plot(self, directory, width = 5, height = 5, dpi = 100):
        path = f'{directory}/{self.name}.png'
        # Visualize the confusion matrix
        plt.figure(figsize=(width, height))
        plt.imshow(self.cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {self.name} \n Accuracy: {self.acc}")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(path, dpi = dpi)
        plt.close()