import json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from google import genai
from google.genai import types

import os
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

tr = pd.read_csv("train.csv")
gold = tr[tr["ACL"].notna()].reset_index(drop=True)
print("gold reports:", len(gold))

SPEC = {
'Lateral OA': """Osteoarthritis of the LATERAL femorotibial compartment specifically.
POSITIVE: lateral joint space narrowing; cartilage thinning or defect over the lateral femoral
condyle or lateral tibial plateau; lateral osteophytes; subchondral sclerosis, cysts or reactive
marrow change in the lateral compartment; a generic statement of tricompartmental osteoarthritis
or advanced gonarthrosis (these include the lateral compartment).
NEGATIVE: osteoarthritis described only in the medial compartment, or only patellofemoral /
retropatellar. Compartment matters -- do not transfer medial findings to lateral.""",

'PF OA': """Osteoarthritis of the PATELLOFEMORAL compartment (patella against trochlea).
POSITIVE: chondromalacia patellae at any grade; thinning, fissuring, fibrillation or full-thickness
loss of patellar or trochlear cartilage; retropatellar or trochlear osteophytes; subchondral marrow
change at a patellar facet or the trochlea; narrowing of the patellofemoral joint; a generic
statement of tricompartmental osteoarthritis.
NEGATIVE: osteoarthritis confined to the femorotibial compartments; patellar maltracking, tilt
or a plica with no cartilage finding; prepatellar bursitis.""",

'Effusion': """Fluid within the knee joint cavity.
POSITIVE: joint effusion of any volume including trace, small or minimal; fluid or distension in
the suprapatellar recess, suprapatellar pouch or bursa; hydrarthrosis; haemarthrosis or lipohaemarthrosis;
synovial fluid described as increased.
NEGATIVE: a Baker's / popliteal cyst with no intra-articular fluid described (that is a separate
finding); bursitis outside the joint such as prepatellar, pes anserine or ITB; soft-tissue oedema;
a report stating no effusion.""",

'Contusion': """Traumatic bone marrow oedema -- a bone bruise.
POSITIVE: bone contusion or bruise; bone marrow oedema or marrow signal change attributed to trauma
or impaction; trabecular microfracture with intact cortex; a pivot-shift or kissing-contusion pattern
at the lateral femoral condyle and posterolateral tibial plateau.
NEGATIVE: marrow oedema described as degenerative, reactive or adjacent to osteoarthritis;
oedema attributed to a stress reaction or osteonecrosis; a frank cortical fracture with no separate
contusion described; soft-tissue bruising only."""
}

HEAD = """You are a musculoskeletal radiologist extracting ONE finding from knee MRI reports.

Reports may be in Greek, Turkish, English or another language. Read them in their own language.
Never lower your confidence because a report is not in English.

FINDING: {label}
{spec}

For each report output a confidence from 0 to 100 that this finding is present in THIS knee.
Calibrate like this:
  95  stated plainly as present
  80  clearly described but hedged, or implied by a finding that entails it
  60  suggestive wording, or present but with the compartment or cause left ambiguous
  40  a detailed report that discusses this region but stops short of the finding
  15  not mentioned, in a report that covers other findings in some detail
   8  not mentioned, in a short or narrowly focused report
   3  explicitly stated as absent or normal
Use the whole range. Do not round everything to 0 or 100.

REPORTS (indexed):
{reports}"""

schema = types.Schema(type=types.Type.ARRAY, items=types.Schema(
    type=types.Type.OBJECT,
    properties={"i": types.Schema(type=types.Type.INTEGER), "p": types.Schema(type=types.Type.NUMBER)},
    required=["i", "p"]))

def call(model, label, reports, temperature=0, tries=3):
    body = "\n\n".join(f'[i={i}]\n{r}' for i, r in enumerate(reports))
    prompt = HEAD.format(label=label, spec=SPEC[label], reports=body)
    for t in range(tries):
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema, temperature=temperature))
            return json.loads(r.text)
        except Exception:
            if t == tries - 1: raise
            time.sleep(8 * (t + 1))

def score(label, model="gemini-3.5-flash-lite", B=25, temperature=0, verbose=True):
    reps = gold["Report"].tolist()
    p = [np.nan] * len(reps)
    for i in range(0, len(reps), B):
        ch = reps[i:i+B]
        items = call(model, label, ch, temperature=temperature)
        for d in items:
            idx = int(d["i"])
            if 0 <= idx < len(ch):
                p[i+idx] = float(d["p"]) / 100
        time.sleep(4.2)
    p = np.array(p, float)
    ok = ~np.isnan(p)
    a = roc_auc_score((gold[label].values[ok] >= 0.5).astype(int), p[ok])
    if verbose:
        print(f"{label:12s} AUC {a:.3f}  ({ok.sum()}/{len(p)} parsed, "
              f"spread {np.nanmin(p):.2f}-{np.nanmax(p):.2f})")
    return a, p

print("=== replication check: Lateral OA ===")
a1, _ = score('Lateral OA', model="gemini-3.1-flash-lite")
print(f"Lateral OA on gemini-3.1-flash-lite: AUC {a1:.3f}")
a2, _ = score('Lateral OA', model="gemini-3.5-flash-lite", temperature=0.3)
print(f"Lateral OA on gemini-3.5-flash-lite temp=0.3: AUC {a2:.3f}")
print(f"original run (temp=0, 3.5-flash-lite): AUC 0.841, baseline 0.812")
