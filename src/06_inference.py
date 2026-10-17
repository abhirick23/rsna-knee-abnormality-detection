#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, time, numpy as np, pandas as pd, cv2, pydicom, torch, torch.nn as nn, timm
from multiprocessing import Pool

ROOT   = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
MD = '/kaggle/input/datasets/abhiroopsarkar/model-b2-pt'
print(os.listdir(MD))                      # confirm model.pt is here, fix MD if not
SPLIT  = 'test'
K, S   = 24, 192
PLANES = ['Sagittal', 'Coronal', 'Axial']
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
BACKBONE, KU = 'efficientnet_b2', 16
dev = 'cuda'
cv2.setNumThreads(0)

ser = pd.read_csv(f'{ROOT}/{SPLIT}_series.csv')
ser = ser[ser.Fluid_Sensitive == 1]
CAND = {k: g.SeriesInstanceUID.tolist() for k, g in ser.groupby(['StudyInstanceUID', 'Anatomical_Plane'])}
UIDS = pd.read_csv(f'{ROOT}/{SPLIT}.csv').StudyInstanceUID.tolist()

# ---- preprocessing: IDENTICAL to training ----
def sort_key(ds):
    try:
        iop = np.array(ds.ImageOrientationPatient, float)
        n = np.cross(iop[:3], iop[3:])
        return float(np.array(ds.ImagePositionPatient, float) @ n)
    except Exception:
        return float(getattr(ds, 'InstanceNumber', 0) or 0)

def load_series(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    keys = [sort_key(pydicom.dcmread(f, stop_before_pixels=True)) for f in files]
    order = np.argsort(keys)
    idx = np.linspace(0, len(files) - 1, K).round().astype(int)
    cache, sl = {}, []
    for i in idx:
        f = files[order[i]]
        if f not in cache:
            img = pydicom.dcmread(f).pixel_array.astype(np.float32)
            if img.ndim == 3: img = img[img.shape[0] // 2]
            cache[f] = img
        sl.append(cache[f])
    vol = np.stack(sl)
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1) * 255
    out = np.zeros((K, S, S), np.uint8)
    for j, im in enumerate(vol):
        h, w = im.shape
        m = max(h, w)
        pad = np.zeros((m, m), np.float32)
        pad[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = im
        out[j] = cv2.resize(pad, (S, S), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return out

def process_study(uid):
    arr = np.zeros((3, K, S, S), np.uint8)
    mask = np.zeros(3, np.uint8)
    for p, plane in enumerate(PLANES):
        sids = CAND.get((uid, plane), [])
        if not sids: continue
        sids = sorted(sids, key=lambda s: -len(os.listdir(f'{ROOT}/{SPLIT}_series/{uid}/{s}')))
        for s in sids:
            try:
                arr[p] = load_series(f'{ROOT}/{SPLIT}_series/{uid}/{s}')
                mask[p] = 1
                break
            except Exception as e:
                print('fail', uid, s, repr(e)[:100])
    return uid, arr, mask

# ---- model: IDENTICAL to training ----
class Net(nn.Module):
    def __init__(s, pretrained=False):
        super().__init__()
        s.bb = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0, in_chans=1)
        D = s.bb.num_features
        s.att = nn.ModuleList([nn.Linear(D, 1) for _ in range(3)])
        s.drop = nn.Dropout(0.3)
        s.head = nn.Linear(3 * D + 3, 12)
    def forward(s, x, m):
        B, P, K_, H, W = x.shape
        f = s.bb(x.reshape(B * P * K_, 1, H, W)).float().reshape(B, P, K_, -1)
        e = []
        for p in range(3):
            w = torch.softmax(s.att[p](f[:, p]).squeeze(-1), 1)
            e.append((w.unsqueeze(-1) * f[:, p]).sum(1))
        e = torch.stack(e, 1) * m.unsqueeze(-1)
        return s.head(s.drop(torch.cat([e.flatten(1), m], 1)))

def prep(x): return (x.float() / 255 - 0.2) / 0.25

model = Net(pretrained=False).to(dev).to(memory_format=torch.channels_last)
model.load_state_dict(torch.load(f'{MD}/model_b2.pt', map_location=dev))
model.eval()
sl_idx = np.linspace(0, 23, KU).round().astype(int)

# ---- run: preprocess a chunk on CPU, predict, repeat ----
t0, preds = time.time(), []
CH = 128
with Pool(os.cpu_count(), maxtasksperchild=64) as pool:
    for i in range(0, len(UIDS), CH):
        res = pool.map(process_study, UIDS[i:i+CH], chunksize=4)
        X = torch.from_numpy(np.stack([r[1][:, sl_idx] for r in res]))
        M = torch.from_numpy(np.stack([r[2] for r in res]).astype(np.float32))
        for j in range(0, len(X), 8):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                p = model(prep(X[j:j+8].to(dev)), M[j:j+8].to(dev)).float().sigmoid().cpu().numpy()
            preds.append(p)
        print(f'{min(i+CH, len(UIDS))}/{len(UIDS)} {time.time()-t0:.0f}s', flush=True)

P = np.concatenate(preds)
sub = pd.DataFrame(P, columns=LABELS); sub.insert(0, 'StudyInstanceUID', UIDS)
sub.to_csv('submission.csv', index=False)
print(sub.shape, sub.isna().sum().sum(), 'NaNs'); print(sub.head())
