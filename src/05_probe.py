#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# In[ ]:


import warnings, timm, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore', category=UserWarning)

PROBE = 'P3'          # 'BASE' | 'P1' | 'P2' | 'P3'   <-- change this each run
BACKBONE, KU, EPOCHS, BS, LR, P = 'efficientnet_b0', 16, 7, 4, 3e-4, 3
PERFIND = (PROBE == 'P1')
TWO5D   = (PROBE == 'P2')
CH      = 3 if TWO5D else 1
dev = 'cuda'; torch.backends.cudnn.benchmark = True

Y_EVAL  = Y.copy()
Y_TRAIN = Y.copy()
if PROBE == 'P3':
    SRC = ['/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
           '/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v2.csv',
           '/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv',
           '/kaggle/input/datasets/lixin73/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv']
    acc = []
    for f in SRC:
        d = pd.read_csv(f)
        d.columns = [d.columns[0]] + [norm(c) or c for c in d.columns[1:]]
        d = d.drop_duplicates(d.columns[0]).set_index(d.columns[0]).reindex(tr.StudyInstanceUID)
        v = d[LBL].values.astype(np.float32)
        if np.nanmax(v) > 1.5: v = v / 100.0
        acc.append(v); print('  loaded', f.split('/')[-1], 'nan', np.isnan(v).sum())
    Y_TRAIN = np.nan_to_num(np.nanmean(acc, 0).astype(np.float32), nan=0.0)
    print('P3 target = mean of', len(acc), 'tables')

class DS(Dataset):
    def __init__(s, idx, train): s.idx, s.train, s.mm = idx, train, {}
    def __len__(s): return len(s.idx)
    def __getitem__(s, j):
        i = s.idx[j]; ci = i // 256
        if ci not in s.mm: s.mm[ci] = np.load(f'{TMP}/x_{ci:04d}.npy', mmap_mode='r')
        sl = (np.sort(np.random.choice(24, KU, replace=False)) if s.train
              else np.linspace(0, 23, KU).round().astype(int))
        v = s.mm[ci][i % 256]
        if TWO5D:
            nb = np.clip(sl[:, None] + np.array([-1, 0, 1]), 0, 23)
            x = np.array(v[:, nb.ravel()]).reshape(P, KU, 3, v.shape[-2], v.shape[-1])
        else:
            x = np.array(v[:, sl])[:, :, None]
        return (torch.from_numpy(x), torch.from_numpy(MASK[i]), torch.from_numpy(Y_TRAIN[i]))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0, in_chans=CH)
        D = s.bb.num_features; s.D = D
        if PERFIND:
            s.att = nn.ModuleList([nn.Linear(D, 12) for _ in range(P)])
            s.W   = nn.Parameter(torch.zeros(12, P * D)); nn.init.normal_(s.W, std=0.01)
            s.b   = nn.Parameter(torch.zeros(12))
            s.mw  = nn.Linear(P, 12)
        else:
            s.att  = nn.ModuleList([nn.Linear(D, 1) for _ in range(P)])
            s.head = nn.Linear(P * D + P, 12)
        s.drop = nn.Dropout(0.3)
    def forward(s, x, m):
        B, Pp, K, C, H, W = x.shape
        f = s.bb(x.reshape(B * Pp * K, C, H, W)).float().reshape(B, Pp, K, -1)
        if PERFIND:
            e = []
            for p in range(P):
                w = torch.softmax(s.att[p](f[:, p]), 1)
                e.append(torch.einsum('bkc,bkd->bcd', w, f[:, p]))
            e = torch.stack(e, 1) * m[:, :, None, None]
            e = s.drop(e.permute(0, 2, 1, 3).reshape(B, 12, P * s.D))
            return (e * s.W).sum(-1) + s.b + s.mw(m)
        e = []
        for p in range(P):
            w = torch.softmax(s.att[p](f[:, p]).squeeze(-1), 1)
            e.append((w.unsqueeze(-1) * f[:, p]).sum(1))
        e = torch.stack(e, 1) * m.unsqueeze(-1)
        return s.head(s.drop(torch.cat([e.flatten(1), m], 1)))

def prep(x): return (x.float() / 255 - 0.2) / 0.25
def aug(x):
    B = x.shape[0]
    x = x * (1 + 0.15*torch.randn(B,1,1,1,1,1,device=x.device)) + 0.15*torch.randn(B,1,1,1,1,1,device=x.device)
    dy, dx = np.random.randint(-10, 11, 2)
    return torch.roll(x, (int(dy), int(dx)), (4, 5))

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

nong = np.where(~gold)[0]
ti, vi = next(iter(KFold(5, shuffle=True, random_state=0).split(nong)))
tr_idx, val_idx = nong[ti], nong[vi]
print(f'PROBE {PROBE} | train {len(tr_idx)} val {len(val_idx)} | in_chans {CH} | perfind {PERFIND}', flush=True)

model = Net().to(dev).to(memory_format=torch.channels_last)
dl = DataLoader(DS(tr_idx, True), batch_size=BS, shuffle=True, num_workers=4,
                drop_last=True, pin_memory=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*len(dl), pct_start=0.1)
scaler = torch.amp.GradScaler('cuda')
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
    pv = macro_auc(predict(model, val_idx), Y_EVAL[val_idx])
    print(f'  ep{ep} loss {tl/len(dl):.4f} val {pv:.4f} {time.time()-t0:.0f}s', flush=True)
torch.save(model.state_dict(), f'/kaggle/working/probe_{PROBE}.pt')
print(f'\n{PROBE} FINAL val {pv:.4f}   (baseline 0.8600)')
