import matplotlib.pyplot as plt
import numpy as np

# Here's a function to plot a confusion matrix using matplotlib

def plot_confusion_matrix(cm, class_labels):
    """
    Plots a confusion matrix using matplotlib with class labels.

    Args:
    cm (array-like): Confusion matrix array.
    class_labels (list): List of class labels.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Confusion matrix from the user's image
# MODEL: Llama-2 trained on climate fever data
#confusion_matrix = np.array([[244, 22, 39], [23, 22, 0], [41, 2, 82]])

# MODEL: Llma-2 trained on fever and climate fever data
#confusion_matrix = np.array([[229,31,45], [17,28,0],[34, 3, 88]])

# MODEK: GPT-4
confusion_matrix = np.array([[233,  34,  38],[ 13,  27,   5],[ 42,   1 , 82]])

# Placeholder class labels
class_labels = ['Not Enough Info', 'Refutes', 'Supports']  # User can replace these with actual class names

# We need to swap the first and the second class (both row and column) to make "Not Enough Info" the middle class

# New order should be: Refutes, Not Enough Info, Supports



# Swap the rows

confusion_matrix[[0, 1]] = confusion_matrix[[1, 0]]



# Swap the columns

confusion_matrix[:, [0, 1]] = confusion_matrix[:, [1, 0]]



# Update class labels to reflect the new order

class_labels = ['Refutes', 'Not Enough Info', 'Supports']

# Plotting the confusion matrix
plot_confusion_matrix(confusion_matrix, class_labels)
plt.show()
