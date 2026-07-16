import os
"""
Phase 4, Step A: Compute average and modal controllability for all 238
subjects from both MIND and SC structural substrates.

Theoretical basis: Gu et al. (2015) Cell; Pasqualetti et al. (2014)
  System:  dx/dt = A x(t) + B u(t)
           A = spectral-normalised MIND or SC matrix  (360x360)
           B = I_360  (all regions independently controllable)

Average controllability (AC)
  W_c(T) = integral 0..T exp(At) BB' exp(A't) dt
  For symmetric A = V Lambda V':
    W_c_diag[i] = sum_j [(exp(2*lambda_j*T) - 1) / (2*lambda_j)] * v_ij^2
  AC[i] = W_c(T)[i,i]   -- ease of driving brain from any state near region i
  Primary time horizon T=3; sensitivity at T=1, T=5.

Modal controllability (MC)
  MC[i] = sum_j (1 - lambda_j^2) * v_ij^2              [Gu et al. 2015 eq. 5]
  High MC -> region pushes brain toward hard-to-reach (high-energy) states.
  MC is independent of T.

Implementation: fully vectorised batch eigendecomposition via np.linalg.eigh
on (238, 360, 360) stacks -- no per-subject loop needed.

Outputs (all in NCT_inputs/controllability/):
  MIND_AC_T1.npy  (238, 360) float32    average controllability, T=1
  MIND_AC_T3.npy  (238, 360) float32    average controllability, T=3  <- primary
  MIND_AC_T5.npy  (238, 360) float32    average controllability, T=5
  MIND_MC.npy     (238, 360) float32    modal controllability (T-independent)
  SC_AC_T1.npy    (238, 360) float32
  SC_AC_T3.npy    (238, 360) float32    <- primary
  SC_AC_T5.npy    (238, 360) float32
  SC_MC.npy       (238, 360) float32
  subject_index.csv            subject ID -> row index mapping
  controllability_summary.csv  per-subject mean AC/MC for QC
"""

import logging, sys, time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

NCT_DIR     = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
MIND_NORM   = NCT_DIR / 'MIND_norm'
SC_NORM     = NCT_DIR / 'SC_norm'
NODE_CSV    = NCT_DIR / 'node_names.csv'
SUBJ_CSV    = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/Twins_240_beh_sheet_complete_all_vars - Sheet1.csv"))

OUT_DIR     = NCT_DIR / 'Controllability'
OUT_DIR.mkdir(exist_ok=True)

T_PRIMARY   = 3
T_SENSITIVITY = [1, 5]
N_NODES     = 360
EIG_EPS     = 1e-10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()])
log = logging.getLogger('NCT_ctrl')


def average_controllability(eigvals, eigvecs, T):
    """
    Vectorised average controllability for N subjects.
    """
    lam2T  = 2.0 * eigvals * T
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff  = np.where(
            np.abs(eigvals) > EIG_EPS,
            np.expm1(lam2T) / (2.0 * eigvals),
            T
        )

    V2 = eigvecs ** 2
    return np.einsum('nj,nij->ni', coeff, V2)


def modal_controllability(eigvals, eigvecs):
    """
    Vectorised modal controllability for N subjects (T-independent).
    MC[n, i] = sum_j (1 - lambda[n,j]^2) * V[n,i,j]^2
    """
    coeff = 1.0 - eigvals ** 2
    V2    = eigvecs ** 2
    return np.einsum('nj,nij->ni', coeff, V2)


def load_stack_generic(directory, subject_ids, tag):
    """Load stack using the actual filename convention."""
    arrays = []
    for sid in subject_ids:
        p = directory / f'{sid}_{tag}_norm.npy'
        if not p.exists():
            raise FileNotFoundError(f'Missing: {p}')
        m = np.load(str(p)).astype(np.float64)
        if m.shape != (N_NODES, N_NODES):
            raise ValueError(f'{p.name}: wrong shape {m.shape}')
        arrays.append(m)
    return np.stack(arrays, axis=0)


def main():
    t0_total = time.time()

    df = pd.read_csv(str(SUBJ_CSV))
    df = df[df['TwinPairID'] != 'Pair41'].dropna(subset=['Subject'])
    subject_ids = [str(int(float(x))) for x in df['Subject']]
    N = len(subject_ids)
    log.info(f'Subjects: {N}')

    pd.DataFrame({'row': range(N), 'subj_id': subject_ids}).to_csv(
        str(OUT_DIR / 'subject_index.csv'), index=False)

    node_df    = pd.read_csv(str(NODE_CSV))
    node_names = node_df['Region'].tolist()
    assert len(node_names) == N_NODES

    log.info('Loading MIND_norm matrices...')
    t0 = time.time()
    MIND = load_stack_generic(MIND_NORM, subject_ids, 'MIND')
    log.info(f'  Loaded: shape={MIND.shape}  [{timedelta(seconds=int(time.time()-t0))}]')

    log.info('Loading SC_norm matrices...')
    t0 = time.time()
    SC = load_stack_generic(SC_NORM, subject_ids, 'SC')
    log.info(f'  Loaded: shape={SC.shape}  [{timedelta(seconds=int(time.time()-t0))}]')

    log.info('Eigendecomposition: MIND  (238 x 360x360)...')
    t0 = time.time()
    mind_eigvals, mind_eigvecs = np.linalg.eigh(MIND)
    log.info(f'  Done [{timedelta(seconds=int(time.time()-t0))}]')

    log.info('Eigendecomposition: SC    (238 x 360x360)...')
    t0 = time.time()
    sc_eigvals, sc_eigvecs = np.linalg.eigh(SC)
    log.info(f'  Done [{timedelta(seconds=int(time.time()-t0))}]')

    del MIND, SC

    all_T = sorted(set([T_PRIMARY] + T_SENSITIVITY))
    log.info(f'Computing AC for T = {all_T}...')

    for T in all_T:
        MIND_AC = average_controllability(mind_eigvals, mind_eigvecs, T)
        SC_AC   = average_controllability(sc_eigvals,   sc_eigvecs,   T)

        np.save(str(OUT_DIR / f'MIND_AC_T{T}.npy'), MIND_AC.astype(np.float32))
        np.save(str(OUT_DIR / f'SC_AC_T{T}.npy'),   SC_AC.astype(np.float32))

    log.info('Computing MC (T-independent)...')
    MIND_MC = modal_controllability(mind_eigvals, mind_eigvecs)
    SC_MC   = modal_controllability(sc_eigvals,   sc_eigvecs)

    np.save(str(OUT_DIR / 'MIND_MC.npy'), MIND_MC.astype(np.float32))
    np.save(str(OUT_DIR / 'SC_MC.npy'),   SC_MC.astype(np.float32))

    MIND_AC_T3 = np.load(str(OUT_DIR / f'MIND_AC_T{T_PRIMARY}.npy'))
    SC_AC_T3   = np.load(str(OUT_DIR / f'SC_AC_T{T_PRIMARY}.npy'))

    summary = pd.DataFrame({
        'subj_id':         subject_ids,
        'MIND_AC_mean':    MIND_AC_T3.mean(axis=1),
        'MIND_AC_std':     MIND_AC_T3.std(axis=1),
        'SC_AC_mean':      SC_AC_T3.mean(axis=1),
        'SC_AC_std':       SC_AC_T3.std(axis=1),
        'MIND_MC_mean':    MIND_MC.mean(axis=1).astype(np.float32),
        'MIND_MC_std':     MIND_MC.std(axis=1).astype(np.float32),
        'SC_MC_mean':      SC_MC.mean(axis=1).astype(np.float32),
        'SC_MC_std':       SC_MC.std(axis=1).astype(np.float32),
    })
    summary.to_csv(str(OUT_DIR / 'controllability_summary.csv'), index=False)

    mind_ac_group = MIND_AC_T3.mean(axis=0)
    mind_mc_group = MIND_MC.mean(axis=0).astype(np.float32)
    ac_mc_corr    = float(np.corrcoef(mind_ac_group, mind_mc_group)[0, 1])
    log.info(f'MIND AC-MC cross-correlation (group mean): r = {ac_mc_corr:.4f}')

    expected = {f'MIND_AC_T{T}.npy': (N, N_NODES) for T in all_T}
    expected.update({f'SC_AC_T{T}.npy': (N, N_NODES) for T in all_T})
    expected['MIND_MC.npy'] = (N, N_NODES)
    expected['SC_MC.npy']   = (N, N_NODES)

    all_ok = True
    for fname, expected_shape in sorted(expected.items()):
        p = OUT_DIR / fname
        if not p.exists():
            all_ok = False
            continue
        arr = np.load(str(p))
        ok  = arr.shape == expected_shape and not np.isnan(arr).any()
        if not ok:
            all_ok = False

    wall = timedelta(seconds=int(time.time() - t0_total))
    log.info(f'{"ALL OUTPUTS OK" if all_ok else "FAILURES DETECTED"}  Wall time: {wall}')
    log.info(f'Outputs -> {OUT_DIR}/')


if __name__ == '__main__':
    main()
