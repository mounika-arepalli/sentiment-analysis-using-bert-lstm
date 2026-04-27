import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# --------------------------------------------
# Load saved files
# --------------------------------------------
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")
y_prob = np.load("y_pred_prob.npy")


# --------------------------------------------
# 1️⃣ CONFUSION MATRIX
# --------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()


# --------------------------------------------
# 2️⃣ ROC CURVE
# --------------------------------------------
y_true_bin = label_binarize(y_true, classes=[0,1,2])

plt.figure()

for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()