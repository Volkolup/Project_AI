import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class MetricsEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names
    
    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i]
            }
        
        return results
    
    def get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def print_metrics(self, metrics):
        print(f"\n{'='*50}")
        print("OVERALL METRICS")
        print(f"{'='*50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        print(f"\n{'='*50}")
        print("PER-CLASS METRICS")
        print(f"{'='*50}")
        for class_name, scores in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall: {scores['recall']:.4f}")
            print(f"  F1-Score: {scores['f1']:.4f}")
    
    def print_confusion_matrix(self, cm):
        print(f"\n{'='*50}")
        print("CONFUSION MATRIX")
        print(f"{'='*50}")
        print(f"{'':15}", end='')
        for name in self.class_names:
            print(f"{name:15}", end='')
        print()
        for i, name in enumerate(self.class_names):
            print(f"{name:15}", end='')
            for j in range(len(self.class_names)):
                print(f"{cm[i][j]:15}", end='')
            print()
