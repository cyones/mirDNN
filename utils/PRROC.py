#!/usr/bin/python
##
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

assert len(sys.argv)==2, "Wrong number of arguments (one expected)"
specie = sys.argv[1]
##
ds = pd.DataFrame({'precision' : [], 'recall' : [], 'Classifier' : []})
methods = ["GBM", "mirDNN", "BLAST", "RF", "miRNAss"]
for m in methods:
    try:
        pos = pd.read_csv(f"predictions/{m}-{specie}-pos.csv", header=None).iloc[:,1]
        neg = pd.read_csv(f"predictions/{m}-{specie}-neg.csv", header=None).iloc[:,1]
    except:
        continue
    preds = np.concatenate([pos, neg])
    label = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])]).astype(int)
    pr, rc, _ = precision_recall_curve(label, preds)
    toapp = pd.DataFrame({'precision' : pr,
                          'recall' : rc,
                          'Classifier' : [m for i in range(len(pr))]})
    ds = ds.append(toapp)
ds['Classifier'] = ds['Classifier'].astype('category')
fig = sns.lineplot(x=ds.recall,
                   y=ds.precision,
                   style=ds.Classifier,
                   hue=ds.Classifier,
                   style_order=methods,
                   hue_order=methods,
                   ).get_figure()

if specie=="cel": lname="Caenorhabditis elegans"
if specie=="ath": lname="Arabidopsis thaliana"
if specie=="aga": lname="Anopheles gambiae"
if specie=="dre": lname="Danio rerio"
if specie=="hsa": lname="Homo sapiens"

plt.title(lname, fontname='Liberation Sans', style='italic')
fig.savefig(f"results/PRROC-{specie}.pdf")
fig.savefig(f"results/PRROC-{specie}.png")
##
