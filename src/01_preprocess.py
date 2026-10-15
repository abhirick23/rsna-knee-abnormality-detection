#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os, cv2, numpy as np, pandas as pd, pydicom
from multiprocessing import Pool
from tqdm.auto import tqdm

# run once if you hit decode errors on JPEG2000 / JPEG-lossless:
# !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm

ROOT   = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
SPLIT  = 'train'          # do 'test' first as a 3-study dry run, then 'train'
K, S   = 24, 192          # slices per series, image size
PLANES = ['Sagittal', 'Coronal', 'Axial']
OUT    = f'/kaggle/working/{SPLIT}_shards'
CHUNK  = 256
os.makedirs(OUT, exist_ok=True)

ser = pd.read_csv(f'{ROOT}/{SPLIT}_series.csv')
ser = ser[ser.Fluid_Sensitive == 1]
CAND = {k: g.SeriesInstanceUID.tolist() for k, g in ser.groupby(['StudyInstanceUID', 'Anatomical_Plane'])}
UIDS = pd.read_csv(f'{ROOT}/{SPLIT}.csv').StudyInstanceUID.tolist()

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
        # pick the series with the most slices
        sids = sorted(sids, key=lambda s: -len(os.listdir(f'{ROOT}/{SPLIT}_series/{uid}/{s}')))
        for s in sids:
            try:
                arr[p] = load_series(f'{ROOT}/{SPLIT}_series/{uid}/{s}')
                mask[p] = 1
                break
            except Exception as e:
                print('fail', uid, s, repr(e)[:100])
    return uid, arr, mask

import time
cv2.setNumThreads(0)

START, END = 0, 2200           # commit A: 0-2200, commit B: 2200-4407
sub = UIDS[START:END]
chunks = [sub[i:i + CHUNK] for i in range(0, len(sub), CHUNK)]
OUT = f'/kaggle/working/{SPLIT}_shards_{START}'
os.makedirs(OUT, exist_ok=True)

with Pool(os.cpu_count(), maxtasksperchild=64) as pool:
    for ci, ch in enumerate(chunks):
        t0 = time.time()
        res = pool.map(process_study, ch, chunksize=4)
        np.savez(f'{OUT}/shard_{ci:04d}.npz',
                 x=np.stack([r[1] for r in res]),
                 m=np.stack([r[2] for r in res]),
                 uid=np.array([r[0] for r in res]))
        print(f"chunk {ci+1}/{len(chunks)} {time.time()-t0:.0f}s", flush=True)

# In[ ]:


import glob, os, zipfile, numpy as np, pandas as pd

ROOT = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
UIDS = pd.read_csv(f'{ROOT}/train.csv').StudyInstanceUID.tolist()
CHUNK, K, S = 256, 24, 192

files = sorted(glob.glob('/kaggle/input/**/shard_*.npz', recursive=True))
print(len(files), "shard files found")

report, seen = [], set()
for f in files:
    name = os.path.basename(f)
    ci = int(name.split('_')[1].split('.')[0])
    row = {'shard': name, 'ok': False}
    try:
        # 1) zip integrity: CRC of every member
        bad = zipfile.ZipFile(f).testzip()
        assert bad is None, f"CRC fail in {bad}"

        d = np.load(f)
        x, m, uid = d['x'], d['m'], d['uid']

        # 2) shapes/dtypes
        exp_n = len(UIDS[ci*CHUNK:(ci+1)*CHUNK])
        assert x.shape == (exp_n, 3, K, S, S) and x.dtype == np.uint8, f"x {x.shape} {x.dtype}"
        assert m.shape == (exp_n, 3), f"m {m.shape}"

        # 3) right studies, right order, no duplicates
        assert list(uid) == UIDS[ci*CHUNK:(ci+1)*CHUNK], "uid mismatch/order"
        assert not (set(uid) & seen), "duplicate uids across shards"
        seen |= set(uid)

        # 4) content sanity
        mask_planes = m.sum(1)
        empty_studies = int((mask_planes == 0).sum())
        # planes flagged present but all-zero pixels = silent failure
        present_blank = 0
        for i in range(exp_n):
            for p in range(3):
                if m[i, p] and x[i, p].max() == 0: present_blank += 1
        # last study (most likely truncated if cancelled mid-write)
        last_ok = x[-1][m[-1] == 1].mean() > 0 if m[-1].any() else True

        row.update(ok=True, n=exp_n, sag=m[:,0].mean().round(3), cor=m[:,1].mean().round(3),
                   axi=m[:,2].mean().round(3), empty_studies=empty_studies,
                   present_blank=present_blank, mean_int=round(float(x.mean()), 1), last_ok=last_ok)
    except Exception as e:
        row['err'] = repr(e)[:120]
    report.append(row)
    print(row)

df = pd.DataFrame(report)
print("\nGOOD:", int(df.ok.sum()), "/", len(df))
print("covered studies:", len(seen), "of", len(UIDS))
