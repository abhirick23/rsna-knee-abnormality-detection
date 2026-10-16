import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

pred = pd.read_json("gold_labels.jsonl", lines=True)
train = pd.read_csv("train.csv")
g = train.dropna(subset=LABELS).merge(pred, on="StudyInstanceUID", suffixes=("", "_p"))
print(len(pred), "labeled |", len(g), "gold overlap")
for k in LABELS:
    if g[k].nunique() > 1:
        print(f"{k:18s} {roc_auc_score(g[k], g[k+'_p']):.3f}")
    else:
        print(f"{k:18s} only one class present ({g[k].iloc[0]}), AUC undefined")
