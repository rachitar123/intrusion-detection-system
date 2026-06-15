import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, classes, title='Confusion matrix', save_path='confusion_matrix.png'):
    print(f"Plotting confusion matrix to {save_path}...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_multiclass_roc(y_true, y_prob, classes, title='Multiclass ROC', save_path='roc_curve.png'):
    print(f"Plotting ROC curves to {save_path}...")
    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown']
    
    for i, color in zip(range(n_classes), colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_distribution(confidences, save_path='confidence_dist.png'):
    print(f"Plotting confidence distribution to {save_path}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=30, kde=True, color='blue', alpha=0.6)
    plt.title('Distribution of Prediction Confidences')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
