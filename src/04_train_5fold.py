#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, re, numpy as np, pandas as pd
from tqdm.auto import tqdm

ROOT = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
DFS  = '/kaggle/input/datasets/abhiroopsarkar/knee-shards-all/shards'
LABF = '/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'
TMP  = '/kaggle/temp/xs'
LBL  = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
        'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
os.makedirs(TMP, exist_ok=True)

ALIAS = {}
def _reg(c, *n):
    for x in (c,) + n: ALIAS[re.sub(r'[^a-z0-9]','',x.lower())] = c
_reg('ACL','acl_tear'); _reg('MCL','mcl_tear')
_reg('Medial Meniscus','medial_meniscus','med_meniscus','medmen')
_reg('Lateral Meniscus','lateral_meniscus','lat_meniscus','latmen')
_reg('Medial OA','medial_oa','med_oa','medoa'); _reg('Lateral OA','lateral_oa','lat_oa','latoa')
_reg('PF OA','pf_oa','pfoa','patellofemoral_oa'); _reg('Effusion','joint_effusion')
_reg('Synovitis'); _reg("Baker's",'bakers','bakers_cyst','baker_cyst','popliteal_cyst')
_reg('Contusion','bone_contusion','bone_bruise'); _reg('Fracture','bone_fracture')
def norm(s): return ALIAS.get(re.sub(r'[^a-z0-9]','',str(s).lower()))

tr = pd.read_csv(f'{ROOT}/train.csv')
rl = pd.read_csv(LABF)
rl.columns = [rl.columns[0]] + [norm(c) or c for c in rl.columns[1:]]
rl = rl.drop_duplicates(rl.columns[0]).set_index(rl.columns[0]).reindex(tr.StudyInstanceUID)
Y  = rl[LBL].values.astype(np.float32)
if np.nanmax(Y) > 1.5: Y = Y / 100.0
assert not np.isnan(Y).any(), 'label NaNs'
G    = tr[LBL].values.astype(np.float32)
gold = ~np.isnan(G).any(1)
print('labels', Y.shape, 'range', Y.min(), Y.max(), '| gold', gold.sum())

masks = []
for ci in tqdm(range(18)):
    a = np.load(f'{DFS}/shard_{ci:04d}.npz')
    np.save(f'{TMP}/x_{ci:04d}.npy', a['x'])
    masks.append(a['m']); del a
MASK = np.concatenate(masks).astype(np.float32)
print('MASK', MASK.shape, MASK.mean(0).round(3))
assert len(MASK) == len(tr)

# In[2]:


import warnings, timm, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore', category=UserWarning)

BACKBONE, KU, EPOCHS, BS, LR, P, NFOLD = 'efficientnet_b0', 16, 7, 4, 3e-4, 3, 5
dev = 'cuda'; torch.backends.cudnn.benchmark = True

class DS(Dataset):
    def __init__(s, idx, train): s.idx, s.train, s.mm = idx, train, {}
    def __len__(s): return len(s.idx)
    def __getitem__(s, j):
        i = s.idx[j]; ci = i // 256
        if ci not in s.mm: s.mm[ci] = np.load(f'{TMP}/x_{ci:04d}.npy', mmap_mode='r')
        sl = (np.sort(np.random.choice(24, KU, replace=False)) if s.train
              else np.linspace(0, 23, KU).round().astype(int))
        return (torch.from_numpy(np.array(s.mm[ci][i % 256][:, sl])),
                torch.from_numpy(MASK[i]), torch.from_numpy(Y[i]))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0, in_chans=1)
        D = s.bb.num_features
        s.att  = nn.ModuleList([nn.Linear(D, 1) for _ in range(P)])
        s.drop = nn.Dropout(0.3)
        s.head = nn.Linear(P * D + P, 12)
    def forward(s, x, m):
        B, Pp, K, H, W = x.shape
        f = s.bb(x.reshape(B*Pp*K, 1, H, W)).float().reshape(B, Pp, K, -1)
        e = [(torch.softmax(s.att[p](f[:,p]).squeeze(-1),1).unsqueeze(-1)*f[:,p]).sum(1)
             for p in range(P)]
        e = torch.stack(e, 1) * m.unsqueeze(-1)
        return s.head(s.drop(torch.cat([e.flatten(1), m], 1)))

def prep(x): return (x.float()/255 - 0.2) / 0.25
def aug(x):
    B = x.shape[0]
    x = x*(1+0.15*torch.randn(B,1,1,1,1,device=x.device)) + 0.15*torch.randn(B,1,1,1,1,device=x.device)
    dy, dx = np.random.randint(-10, 11, 2)
    return torch.roll(x, (int(dy), int(dx)), (3, 4))

def macro_auc(Pm, T):
    T = T >= 0.5
    return float(np.mean([roc_auc_score(T[:,k], Pm[:,k]) for k in range(12)
                          if 0 < T[:,k].sum() < len(T)]))

@torch.no_grad()
def predict(model, idx):
    model.eval(); out = []
    for x, m, _ in DataLoader(DS(idx, False), batch_size=6, num_workers=4):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(prep(x.to(dev)), m.to(dev)).float().sigmoid().cpu())
    return torch.cat(out).numpy()

gold_idx = np.where(gold)[0]
nong     = np.where(~gold)[0]
oof      = np.zeros((len(Y), 12), np.float32)
golds    = []

for fo, (ti, vi) in enumerate(KFold(NFOLD, shuffle=True, random_state=0).split(nong)):
    tr_idx, val_idx = nong[ti], nong[vi]
    print(f'\n--- fold {fo} | train {len(tr_idx)} val {len(val_idx)} ---', flush=True)
    model = Net().to(dev).to(memory_format=torch.channels_last)
    dl = DataLoader(DS(tr_idx, True), batch_size=BS, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                total_steps=EPOCHS*len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda'); best = 0
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tl = 0
        for x, m, y in dl:
            x, m, y = aug(prep(x.to(dev))), m.to(dev), y.to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        pv = macro_auc(predict(model, val_idx), Y[val_idx])
        print(f'  ep{ep} loss {tl/len(dl):.4f} val {pv:.3f} {time.time()-t0:.0f}s', flush=True)
        if pv > best:
            best = pv; torch.save(model.state_dict(), f'/kaggle/working/fold{fo}.pt')
    model.load_state_dict(torch.load(f'/kaggle/working/fold{fo}.pt'))
    oof[val_idx] = predict(model, val_idx)
    golds.append(predict(model, gold_idx))
    print(f'  fold {fo} gold {macro_auc(golds[-1], G[gold_idx]):.4f}', flush=True)
    del model; torch.cuda.empty_cache()

np.save('/kaggle/working/oof.npy', oof)
Pg = np.mean(golds, 0); np.save('/kaggle/working/gold_5fold.npy', Pg)
print('\nOOF macro   :', round(macro_auc(oof[nong], Y[nong]), 4))
print('5-fold GOLD :', round(macro_auc(Pg, G[gold_idx]), 4))
