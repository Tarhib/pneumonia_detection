from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

def check_metric(predictions, true_labels):
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Generate classification report for per-class metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Display classification report and confusion matrix
    print("\nClassification Report:\n", classification_report(true_labels, predictions))
    print("\nConfusion Matrix:\n", cm)
    
    # Convert the classification report to a DataFrame
    metrics_df = pd.DataFrame(report).transpose()
    
    # Add overall accuracy as a separate row
    accuracy = np.trace(cm) / np.sum(cm)
    metrics_df.loc['accuracy_all'] = [accuracy, None, None, None]
    
    return metrics_df

    

