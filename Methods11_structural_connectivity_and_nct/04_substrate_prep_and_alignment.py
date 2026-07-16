"""
Step 2: NCT substrate preparation -- extract, align, normalise.

For each of 238 subjects:
  1. Load MIND matrix        360x360 .npy (corrThickness, HCP-MMP)
  2. Load SC matrix          410x410 .csv (SIFT2-weighted)
  3. Extract cortical SC     rows/cols 50-409  ->  360x360 (labels 51-410)
  4. Align SC -> MIND order   reorder by region name match
  5. Spectral-normalise both A_norm = A / (1 + lambda_max(A))  ->  rho < 1 strictly
  6. Symmetry + stability QC
  7. Save float32 .npy

One-time outputs:
  NCT_inputs/node_names.csv      360 region names in shared ordering
  NCT_inputs/sc_align_idx.npy   permutation index SC_cortical -> MIND order

Per-subject outputs:
  NCT_inputs/MIND_norm/{subj}_MIND_norm.npy   float32 360x360
  NCT_inputs/SC_norm/{subj}_SC_norm.npy       float32 360x360

Atlas label scheme used for extraction:
  SC rows/cols 0-49   (labels 1-50)   Tian S3 subcortical  <- discarded here
  SC rows/cols 50-229 (labels 51-230) HCP-MMP RH cortical  <- kept
  SC rows/cols 230-409(labels 231-410)HCP-MMP LH cortical  <- kept

Parallelised: ProcessPoolExecutor N_WORKERS=40
"""

import os, sys, time, argparse, logging, traceback
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nibabel as nib

MIND_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "MIND_matrices"))
SC_DIR     = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "SC_matrices"))
HUBS_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "GroupAvgValidation/Group_MIND_Nodal_Degrees.csv"))
ATLAS_PATH = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "Atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S3.dlabel.nii"))
SUBJ_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/Twins_240_beh_sheet_complete_all_vars - Sheet1.csv"))

NCT_DIR    = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
MIND_NORM  = NCT_DIR / 'MIND_norm'
SC_NORM    = NCT_DIR / 'SC_norm'

N_WORKERS  = 40
N_NODES    = 360
SC_FULL    = 410
SC_OFFSET  = 50

for d in [NCT_DIR, MIND_NORM, SC_NORM]:
    d.mkdir(parents=True, exist_ok=True)


def build_alignment(mind_names, atlas_path):
    """
    Build permutation index that reorders SC cortical submatrix to match
    the MIND node ordering, matched by region name string identity.
    """
    img = nib.load(str(atlas_path))
    ax  = img.header.get_axis(0)
    tbl = ax.label[0]

    sc_names = []
    for i in range(N_NODES):
        label_int = SC_OFFSET + 1 + i
        entry = tbl.get(label_int)
        if entry is None:
            raise KeyError(f'Label {label_int} missing from atlas label table')
        name = entry[0] if isinstance(entry, tuple) else str(entry)
        sc_names.append(name)

    sc_name_to_idx = {name: i for i, name in enumerate(sc_names)}

    missing = [n for n in mind_names if n not in sc_name_to_idx]
    if missing:
        raise ValueError(
            f'{len(missing)}/360 MIND region names not found in SC atlas '
            f'label table. First 5 missing: {missing[:5]}')

    align_idx = np.array([sc_name_to_idx[n] for n in mind_names], dtype=np.int32)
    return align_idx, sc_names


def spectral_normalise(A):
    """
    Enforce symmetry and normalise for NCT stability.
    A_norm = A_sym / (1 + lambda_max(A_sym))   ->   rho(A_norm) < 1 strictly.
    """
    A_sym   = (A + A.T) * 0.5
    lam_max = float(np.linalg.eigvalsh(A_sym).max())
    A_norm  = (A_sym / (1.0 + lam_max)).astype(np.float32)
    rho     = float(np.linalg.eigvalsh(A_norm).max())
    return A_norm, lam_max, rho


def process_subject(subj_id, align_idx):
    out_mind = MIND_NORM / f'{subj_id}_MIND_norm.npy'
    out_sc   = SC_NORM   / f'{subj_id}_SC_norm.npy'

    if out_mind.exists() and out_sc.exists():
        return {'subj': subj_id, 'status': 'SKIPPED',
                'lam_m': None, 'lam_s': None,
                'rho_m': None, 'rho_s': None, 'elapsed': 0.0}

    t0 = time.time()
    try:
        mp = MIND_DIR / f'{subj_id}_MIND.npy'
        if not mp.exists():
            raise FileNotFoundError(f'MIND not found: {mp.name}')
        M = np.load(str(mp)).astype(np.float64)
        if M.shape != (N_NODES, N_NODES):
            raise ValueError(f'MIND shape {M.shape} != ({N_NODES},{N_NODES})')

        sp = SC_DIR / f'{subj_id}_SC_SIFT2_410.csv'
        if not sp.exists():
            raise FileNotFoundError(f'SC not found: {sp.name}')
        SC_full = np.loadtxt(str(sp), delimiter=',')
        if SC_full.shape != (SC_FULL, SC_FULL):
            raise ValueError(f'SC shape {SC_full.shape} != ({SC_FULL},{SC_FULL})')

        SC_ctx = SC_full[SC_OFFSET:, SC_OFFSET:].copy()

        SC_ctx = SC_ctx[np.ix_(align_idx, align_idx)]

        M_norm,  lam_m, rho_m = spectral_normalise(M)
        SC_norm, lam_s, rho_s = spectral_normalise(SC_ctx)

        if rho_m >= 1.0:
            raise ValueError(f'rho(MIND_norm)={rho_m:.8f} >= 1 -- stability violated')
        if rho_s >= 1.0:
            raise ValueError(f'rho(SC_norm)={rho_s:.8f} >= 1 -- stability violated')

        if M_norm.min() < -1e-6:
            raise ValueError(f'Negative MIND_norm values: min={M_norm.min():.4e}')
        if SC_norm.min() < -1e-6:
            raise ValueError(f'Negative SC_norm values: min={SC_norm.min():.4e}')

        np.save(str(out_mind), M_norm)
        np.save(str(out_sc),   SC_norm)

        return {'subj': subj_id, 'status': 'OK',
                'lam_m': round(lam_m, 4),
                'lam_s': round(lam_s, 4),
                'rho_m': round(rho_m, 8),
                'rho_s': round(rho_s, 8),
                'elapsed': round(time.time() - t0, 2)}

    except Exception as exc:
        return {'subj': subj_id,
                'status': f'FAILED: {exc}\n{traceback.format_exc()}',
                'lam_m': None, 'lam_s': None,
                'rho_m': None, 'rho_s': None,
                'elapsed': round(time.time() - t0, 2)}


def main():
    parser = argparse.ArgumentParser(description='NCT substrate preparation')
    parser.add_argument('--dry-run',  action='store_true')
    parser.add_argument('--subject',  default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()])
    log = logging.getLogger('NCT_prep')

    df = pd.read_csv(str(SUBJ_CSV))
    df = df[df['TwinPairID'] != 'Pair41'].dropna(subset=['Subject'])
    subjects = [str(int(float(x))) for x in df['Subject']]
    log.info(f'Subjects: {len(subjects)}')

    log.info('Building SC -> MIND node alignment...')
    hubs = pd.read_csv(str(HUBS_CSV))
    mind_names = hubs['Region'].tolist()
    if len(mind_names) != N_NODES:
        log.error(f'Expected {N_NODES} MIND regions, got {len(mind_names)}')
        sys.exit(1)

    align_idx, sc_names = build_alignment(mind_names, ATLAS_PATH)

    assert len(set(align_idx.tolist())) == N_NODES, \
        'FATAL: alignment index has duplicate entries -- region name collision'
    assert int(align_idx.min()) >= 0 and int(align_idx.max()) < N_NODES, \
        'FATAL: alignment index out of bounds'
    log.info(f'Alignment: all {N_NODES}/360 region names matched')

    np.save(str(NCT_DIR / 'sc_align_idx.npy'), align_idx)
    node_df = pd.DataFrame({
        'Position':       range(N_NODES),
        'Region':         mind_names,
        'SC_source_idx':  align_idx.tolist(),
        'SC_atlas_label': [SC_OFFSET + 1 + int(align_idx[j]) for j in range(N_NODES)],
    })
    node_df.to_csv(str(NCT_DIR / 'node_names.csv'), index=False)
    log.info(f'Node order saved -> {NCT_DIR}/node_names.csv')

    if args.subject:
        subjects = [args.subject]
    elif args.dry_run:
        subjects = subjects[:1]

    already_done = [s for s in subjects
                    if (MIND_NORM / f'{s}_MIND_norm.npy').exists() and
                       (SC_NORM   / f'{s}_SC_norm.npy').exists()]
    to_run = [s for s in subjects if s not in already_done]
    log.info(f'To run: {len(to_run)}   Already done: {len(already_done)}')

    t_start = time.time()
    results = []

    if len(to_run) <= 1:
        for s in to_run:
            r = process_subject(s, align_idx)
            results.append(r)
    else:
        n_total = len(to_run)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(process_subject, s, align_idx): s
                       for s in to_run}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)

    for s in already_done:
        results.append({'subj': s, 'status': 'SKIPPED',
                        'lam_m': None, 'lam_s': None,
                        'rho_m': None, 'rho_s': None, 'elapsed': 0.0})

    ok_r   = [r for r in results if r['status'] == 'OK']
    skip_r = [r for r in results if r['status'] == 'SKIPPED']
    fail_r = [r for r in results if r['status'].startswith('FAILED')]
    wall   = timedelta(seconds=int(time.time() - t_start))

    log.info(f'OK: {len(ok_r)}  Skipped: {len(skip_r)}  Failed: {len(fail_r)}  Wall: {wall}')

    if ok_r:
        audit_df = pd.DataFrame([
            {'subj': r['subj'], 'lam_max_mind': r['lam_m'],
             'lam_max_sc': r['lam_s'], 'rho_mind': r['rho_m'],
             'rho_sc': r['rho_s']}
            for r in ok_r
        ])
        audit_df.to_csv(str(NCT_DIR / 'normalisation_audit.csv'), index=False)

    if fail_r:
        for r in fail_r:
            log.error(f'  {r["subj"]}:\n{r["status"]}')

    log.info(f'Outputs -> {NCT_DIR}/')
    log.info('Done. Next step: nct_controllability.py')


if __name__ == '__main__':
    main()
