import pandas as pd
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
pred = pd.read_json("gold_labels.jsonl", lines=True)
train = pd.read_csv("train.csv")
g = train.dropna(subset=LABELS).merge(pred, on="StudyInstanceUID", suffixes=("", "_p"))
g["err"] = (g["Synovitis"] - g["Synovitis_p"]).abs()
for _, r in g.sort_values("err", ascending=False).head(5).iterrows():
    print("gold:", r["Synovitis"], "pred:", r["Synovitis_p"])
    report = r["Report"]
    mentions = "synovi" in report.lower()
    print("mentions 'synovi' anywhere in report:", mentions)
    if mentions:
        idx = report.lower().index("synovi")
        print("context:", report[max(0,idx-200):idx+200].encode("ascii","replace").decode())
    print("---")
