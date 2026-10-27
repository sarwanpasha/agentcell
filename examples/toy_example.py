#!/usr/bin/env python3
"""Self-contained example of the scoring used throughout AgentCell. Needs only numpy.

It builds synthetic treatment profiles with the structure of BBBC021 (compounds, MoAs, batches and
three channels), scores them with NSC and NSCB under plain concatenation and per-channel L2
normalization, checks that scoring from per-channel Gram matrices gives exactly the same result as
scoring from the features, and evaluates one configuration on held-out compounds.

Run: python examples/toy_example.py
"""
import numpy as np

rng = np.random.default_rng(0)
N_MOA, COMP_PER_MOA, DOSES, N_BATCH, DIM = 6, 3, 2, 4, 64
CH = ["DNA", "tubulin", "actin"]

# ---------------------------------------------------------------- synthetic treatments
moa, comp, batch = [], [], []
for m in range(N_MOA):
    for c in range(COMP_PER_MOA):
        for d in range(DOSES):
            moa.append(m); comp.append(m * COMP_PER_MOA + c); batch.append(rng.integers(N_BATCH))
moa, comp, batch = map(np.array, (moa, comp, batch)); N = len(moa)
# each channel carries MoA signal of a different strength plus a batch effect and noise
strength = {"DNA": 1.0, "tubulin": 1.6, "actin": 0.6}
moa_centers = {c: rng.normal(size=(N_MOA, DIM)) for c in CH}
batch_shift = {c: rng.normal(size=(N_BATCH, DIM)) for c in CH}
X = {c: strength[c] * moa_centers[c][moa] + 0.8 * batch_shift[c][batch] + rng.normal(size=(N, DIM)) for c in CH}


# ---------------------------------------------------------------- scoring
def nsc(S, use_batch, queries=None):
    """Fraction of queries whose nearest admissible neighbor shares the MoA (NSC or NSCB)."""
    hits = []
    for i in (range(N) if queries is None else np.where(queries)[0]):
        ok = comp != comp[i]
        if use_batch:
            ok &= batch != batch[i]
        if ok.any():
            j = np.where(ok)[0][np.argmax(S[i, ok])]
            hits.append(moa[j] == moa[i])
    return float(np.mean(hits))


def cos(A):
    A = A / np.linalg.norm(A, axis=1, keepdims=True); return A @ A.T


def sim_from_features(norm):
    if norm == "plain":
        return cos(np.hstack([X[c] for c in CH]))
    return cos(np.hstack([X[c] / np.linalg.norm(X[c], axis=1, keepdims=True) for c in CH]))


def sim_from_grams(G, norm):
    if norm == "plain":
        P = sum(G.values()); d = np.sqrt(np.diag(P)); return P / np.outer(d, d)
    return sum(g / np.outer(np.sqrt(np.diag(g)), np.sqrt(np.diag(g))) for g in G.values()) / len(G)


G = {c: X[c] @ X[c].T for c in CH}          # what embed_ckpts.py stores per channel
print(f"{N} treatments, {len(set(comp))} compounds, {N_MOA} MoAs, {N_BATCH} batches")
for norm in ("plain", "l2"):
    Sf, Sg = sim_from_features(norm), sim_from_grams(G, norm)
    assert np.allclose(Sf, Sg), "Gram decomposition must reproduce direct scoring"
    print(f"{norm:5s}  NSC {nsc(Sf, False):.3f}  NSCB {nsc(Sf, True):.3f}  (Gram scoring identical)")
for c in CH:
    print(f"single channel {c:8s} NSC {nsc(cos(X[c]), False):.3f}")

# ---------------------------------------------------------------- held-out compounds
sel = []
for m in range(N_MOA):
    cs = sorted(set(comp[moa == m])); rng.shuffle(cs); sel += cs[:(len(cs) + 1) // 2]
S_half = np.isin(comp, sel)
S = sim_from_features("l2")
print(f"per-channel L2 on the selection half {nsc(S, False, S_half):.3f}, on the test half {nsc(S, False, ~S_half):.3f}")
