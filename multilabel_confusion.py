import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_excel("output.xlsx")
true = []
pred = []

for ind, row in df.iterrows():
    true.append([row["Hospital (NPE)"], row["Malaria (NPE)"], row["Farming (NPE)"], row["School (NPE)"]])
    tmp = [row["test_npe_hospital"], row["test_npe_malaria"], row["test_npe_farming"], row["test_npe_school"]]
    pred.append([min(1, x) for x in tmp])

true = np.array(true)
pred = np.array(pred)

labels = ["hospital", "malaria", "farming", "school"]
conf_mat_dict = {}

for label_col in range(len(labels)):
    true_label = true[:, label_col]
    pred_label = pred[:, label_col]
    conf_mat_dict[labels[label_col]] = metrics.confusion_matrix(y_pred=pred_label, y_true=true_label)

f, axes = plt.subplots(1, 4, figsize=(15, 5))
axes = axes.ravel()
i = 0

for key, conf_mat in conf_mat_dict.items():
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat)
    disp.plot(ax=axes[i], values_format = '.4g', cmap=plt.cm.Purples)

    if i > 0:
        disp.ax_.set_ylabel("")

    i += 1
    disp.ax_.set_title(key)
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.show()