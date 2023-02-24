import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

data = pd.read_csv('../Results/RIM_Oneshot_Tuned61_CV/Config_1/predictions_2.csv')

df = pd.DataFrame(data)
df.columns = ['file', 'label', 'prediction']
print(df.head(10))
df.info()
print(type(data))
labels = df['label']
predictions = df['prediction']

# Compute precision and recall at different thresholds
precision, recall, thresholds = precision_recall_curve(labels, predictions)

# Plot the precision-recall curve
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
print(best_threshold)

precision_threshold = None
for i in range(len(thresholds)):
    if precision[i] == 1.0 and (precision_threshold is None or recall[i] > recall[precision_threshold]):
        precision_threshold = i

if precision_threshold is not None:
    chosen_threshold = thresholds[precision_threshold]
print(chosen_threshold)