#!/usr/bin/python
##
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
##
ds = pd.DataFrame({'AUCPR' : [], 'Classifier' : [], 'Species' : []})
##
methods = ["mirDNN", "BLAST", "RF", "miRNAss"]
for m in methods:
    for s in ["aga", "ath", "cel", "dre", "hsa"]:
        try:
            pos = pd.read_csv(f"predictions/{m}-{s}-pos.csv", header=None).iloc[:,1]
            neg = pd.read_csv(f"predictions/{m}-{s}-neg.csv", header=None).iloc[:,1]
        except:
            continue
        preds = np.concatenate([pos, neg])
        label = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])]).astype(int)
        pr, rc, _ = precision_recall_curve(label, preds)
        prauc = auc(rc, pr)
        ds = ds.append({'AUCPR':prauc, 'Classifier':m, 'Species':s}, ignore_index=True)
##
ds['Classifier'] = ds['Classifier'].astype('category')
ds['Species'] = ds['Species'].astype('category')
##
fig = sns.barplot(x=ds.Species,
                  y=ds.AUCPR,
                  hue=ds.Classifier,
                  hue_order=methods).get_figure()
plt.title("Method comparison")
fig.savefig("results/barplot.pdf")
fig.savefig("results/barplot.png")
##
print(ds)
##
