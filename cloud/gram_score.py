#!/usr/bin/env python3
"""Score a seed whose 3 channels live on different machines, using per-channel Gram
matrices (103x103, treatment-median features) instead of moving feature CSVs.
  export TAG FEATDIR CH OUTFILE      -> writes packed Gram (int16 cosines + float32 norms) + treatment order
  combine TAG FEATDIR LOCAL_CHS GRAMFILE...  -> NSC/NSCB plain and per_channel_l2
Exactly equivalent to tool_score: cosine of concatenated blocks = sum of block Grams."""
import sys, json, base64, numpy as np
import os
sys.path.insert(0, os.environ['AGENTCELL_TOOLS'])
import tool_score as ts

def gram(tag, d, ch):
    df, keep = ts.load_channel(tag, ch, d)
    df = df.sort_values('treatment').reset_index(drop=True)
    F = df[keep].to_numpy(float)
    return F @ F.T, df

def acc(S, comp, moa, batch=None):
    t = []
    for i in range(len(comp)):
        m = comp != comp[i]
        if batch is not None: m &= batch != batch[i]
        if not m.any(): continue
        j = np.where(m)[0][int(np.argmax(S[i, m]))]
        t.append(moa[j] == moa[i])
    return round(float(np.mean(t)), 4), len(t)

def pack(G):
    n = np.sqrt(np.diag(G)); C = G / np.outer(n, n); iu = np.triu_indices(len(G), 1)
    q = np.round(np.clip(C[iu], -1, 1) * 32767).astype(np.int16)
    return {'q': base64.b64encode(q.tobytes()).decode(), 'norm': base64.b64encode(n.astype(np.float32).tobytes()).decode()}

def unpack(j):
    nn = j['n']; n = np.frombuffer(base64.b64decode(j['norm']), dtype=np.float32).astype(np.float64)
    q = np.frombuffer(base64.b64decode(j['q']), dtype=np.int16).astype(np.float64) / 32767
    C = np.eye(nn); iu = np.triu_indices(nn, 1); C[iu] = q; C.T[iu] = q
    return C * np.outer(n, n)

if sys.argv[1] == 'export':
    tag, d, ch, out = sys.argv[2:6]
    G, df = gram(tag, d, ch)
    json.dump({'ch': ch, 'treat': df['treatment'].astype(int).tolist(), 'n': len(df),
               **pack(G)}, open(out, 'w'))
    print('exported', ch, G.shape, 'trace', float(np.trace(G)))
else:
    tag, d, local = sys.argv[2:5]; files = sys.argv[5:]
    Gs = {}; base = None
    for ch in local.split(','):
        G, df = gram(tag, d, ch); Gs[ch] = G; base = df
    for f in files:
        j = json.load(open(f))
        assert j['treat'] == base['treatment'].astype(int).tolist(), 'treatment order mismatch'
        Gs[j['ch']] = unpack(j)
    assert set(Gs) == {'DAPI', 'Tubulin', 'Actin'}, Gs.keys()
    comp, moa, batch = (base[c].to_numpy() for c in ('compound', 'moa', 'batch'))
    out = {}
    P = sum(Gs.values()); dg = np.sqrt(np.diag(P)); Sp = P / np.outer(dg, dg)
    Sl = sum(G / np.outer(np.sqrt(np.diag(G)), np.sqrt(np.diag(G))) for G in Gs.values())
    for name, S in (('plain', Sp), ('per_channel_l2', Sl)):
        out[name] = {'NSC': acc(S, comp, moa)[0], 'NSCB': acc(S, comp, moa, batch)[0]}
    print('GATHERED', tag, json.dumps(out))
    json.dump(out, open(f'{d}/../score_gathered.json', 'w'))
