"""
Compute MIND (Morphometric INverse Divergence) similarity matrices for all
238 HCP-S1200 twin subjects directly from HCP pipeline CIFTI/GIFTI outputs.

Method: Sebenius et al. (2023) Nature Neuroscience.
Surface space: fsLR 32k (standardised across all HCP subjects).
Parcellation: HCP-MMP v1.0 (Glasser et al. 2016), 360 parcels.
Features: thickness (CT), midthickness vertex area (SA),
          Vol = CT x SA, curvature (MC), sulc (SD).
          All z-scored globally across all valid vertices before KL estimation.

Output per subject:
  {OUT_DIR}/{subj_id}_MIND.npy   float32 360x360 symmetric matrix
  {OUT_DIR}/{subj_id}_MIND.csv   same, with region-name headers
"""

import os, sys, time, argparse, logging, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# CONFIGURATION
DATA_ROOT  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "Raw"))
ATLAS_PATH = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "Atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"))
SUBJ_TABLE = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/Twins_240_beh_sheet_complete_all_vars - Sheet1.csv"))
OUT_DIR    = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "MIND_matrices"))
LOG_DIR    = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "Logs"))

N_WORKERS  = 75   
K_NN       = 1
MIN_VERTS  = 2
N_REGIONS  = 360

for _d in [OUT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

def get_logger(subj_id):
    log_path = LOG_DIR / f'{subj_id}_MIND.log'
    log = logging.getLogger(f'MIND.{subj_id}')
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-5s %(message)s', datefmt='%H:%M:%S'))
        log.addHandler(fh)
    return log


def load_atlas(atlas_path):
    img   = nib.load(str(atlas_path))
    data  = np.asarray(img.dataobj, dtype=np.float32).squeeze()
    parcel_labels = data.astype(np.int32)                        

    label_map = {}
    for i in range(img.ndim):
        ax = img.header.get_axis(i)
        if hasattr(ax, 'label'):
            tbl = ax.label[0]          
            for k, v in tbl.items():
                if k != 0:
                    label_map[k] = v[0] if isinstance(v, tuple) else str(v)
            break

    lh_idx = rh_idx = None
    for i in range(img.ndim):
        ax = img.header.get_axis(i)
        if hasattr(ax, 'iter_structures'):
            for struct_name, slc, bm in ax.iter_structures():
                if 'CORTEX_LEFT'  in struct_name:
                    lh_idx = bm.vertex.copy()
                elif 'CORTEX_RIGHT' in struct_name:
                    rh_idx = bm.vertex.copy()
            break

    if lh_idx is None or rh_idx is None:
        raise RuntimeError('Could not extract LH/RH vertex indices from atlas CIFTI.')

    unique_regions = np.sort(np.unique(parcel_labels[parcel_labels > 0])).astype(np.int32)
    if len(unique_regions) != N_REGIONS:
        raise RuntimeError(
            f'Atlas has {len(unique_regions)} non-zero regions; expected {N_REGIONS}.')

    region_names = [label_map.get(int(r), f'region_{r}') for r in unique_regions]

    return parcel_labels, lh_idx, rh_idx, unique_regions, region_names


def load_cifti_hemi_split(path, lh_idx, rh_idx):
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32).squeeze()  

    lh_data = rh_data = None
    for i in range(img.ndim):
        ax = img.header.get_axis(i)
        if hasattr(ax, 'iter_structures'):
            for struct_name, slc, _ in ax.iter_structures():
                if   'CORTEX_LEFT'  in struct_name:
                    lh_data = data[slc]
                elif 'CORTEX_RIGHT' in struct_name:
                    rh_data = data[slc]
            break

    if lh_data is None or rh_data is None:
        raise RuntimeError(f'Missing LH or RH cortex in {path.name}')

    return lh_data, rh_data


def load_subject_features(subj_id, lh_idx, rh_idx):
    base   = DATA_ROOT / subj_id
    mni32k = base / 'MNINonLinear' / 'fsaverage_LR32k'
    t1w32k = base / 'T1w' / 'fsaverage_LR32k'

    ct_lh, ct_rh = load_cifti_hemi_split(
        mni32k / f'{subj_id}.corrThickness_MSMAll.32k_fs_LR.dscalar.nii',
        lh_idx, rh_idx)
    mc_lh, mc_rh = load_cifti_hemi_split(
        mni32k / f'{subj_id}.curvature_MSMAll.32k_fs_LR.dscalar.nii',
        lh_idx, rh_idx)
    sd_lh, sd_rh = load_cifti_hemi_split(
        mni32k / f'{subj_id}.sulc_MSMAll.32k_fs_LR.dscalar.nii',
        lh_idx, rh_idx)

    sa_lh_full = nib.load(str(
        t1w32k / f'{subj_id}.L.midthickness_MSMAll_va.32k_fs_LR.shape.gii')
        ).darrays[0].data.astype(np.float32)
    sa_rh_full = nib.load(str(
        t1w32k / f'{subj_id}.R.midthickness_MSMAll_va.32k_fs_LR.shape.gii')
        ).darrays[0].data.astype(np.float32)
    sa_lh = sa_lh_full[lh_idx]
    sa_rh = sa_rh_full[rh_idx]

    vol_lh = ct_lh * sa_lh
    vol_rh = ct_rh * sa_rh

    features = np.column_stack([
        np.concatenate([ct_lh,  ct_rh]),
        np.concatenate([sa_lh,  sa_rh]),
        np.concatenate([vol_lh, vol_rh]),
        np.concatenate([mc_lh,  mc_rh]),
        np.concatenate([sd_lh,  sd_rh]),
    ]).astype(np.float32)

    return features


def kl_knn(X, Y, k=1):
    n, d = X.shape
    m    = Y.shape[0]

    tree_x = cKDTree(X)
    r_all, _ = tree_x.query(X, k=k + 1, workers=1)
    r_k = r_all[:, k].astype(np.float64)         

    tree_y = cKDTree(Y)
    s_all, _ = tree_y.query(X, k=k, workers=1)
    s_k = (s_all if s_all.ndim == 1 else s_all[:, k - 1]).astype(np.float64)

    r_k = np.maximum(r_k, 1e-12)
    s_k = np.maximum(s_k, 1e-12)

    kl = -(d / n) * np.sum(np.log(r_k / s_k)) + np.log(m / (n - 1))
    return float(max(kl, 0.0))


def compute_mind_matrix(features, parcel_labels, unique_regions,
                         k=K_NN, min_verts=MIN_VERTS, logger=None):
    n_r  = len(unique_regions)
    mind = np.full((n_r, n_r), np.nan, dtype=np.float32)
    np.fill_diagonal(mind, 1.0)

    region_data = {}
    degenerate = []
    for r in unique_regions:
        mat = features[parcel_labels == r]
        region_data[int(r)] = mat
        if len(mat) < min_verts:
            degenerate.append(int(r))

    if degenerate and logger:
        logger.warning(f'{len(degenerate)} degenerate regions (< {min_verts} verts): '
                       f'{degenerate[:10]}')

    n_pairs = n_r * (n_r - 1) // 2
    done    = 0

    for i in range(n_r):
        r1 = int(unique_regions[i])
        X  = region_data[r1]
        if len(X) < min_verts:
            continue

        for j in range(i + 1, n_r):
            r2 = int(unique_regions[j])
            Y  = region_data[r2]
            if len(Y) < min_verts:
                continue

            kl_xy = kl_knn(X, Y, k=k)
            kl_yx = kl_knn(Y, X, k=k)
            D     = kl_xy + kl_yx          
            val   = np.float32(1.0 / (1.0 + D))
            mind[i, j] = val
            mind[j, i] = val

            done += 1
            if logger and done % 5000 == 0:
                logger.debug(f'  {done}/{n_pairs} pairs ({100*done/n_pairs:.1f}%)')

    return mind


def process_subject(subj_id, parcel_labels, lh_idx, rh_idx,
                    unique_regions, region_names):
    out_npy = OUT_DIR / f'{subj_id}_MIND.npy'
    out_csv = OUT_DIR / f'{subj_id}_MIND.csv'

    if out_npy.exists() and out_csv.exists():
        return subj_id, 'SKIPPED', 0.0, 0

    log = get_logger(subj_id)
    t0  = time.time()

    try:
        log.info(f'=== MIND pipeline starting: {subj_id} ===')

        features = load_subject_features(subj_id, lh_idx, rh_idx)
        n_total  = len(features)
        log.info(f'Features loaded:  shape={features.shape}')

        valid   = (features[:, 0] > 0) & (features[:, 1] > 0) & (features[:, 2] > 0)
        n_valid = int(valid.sum())
        log.info(f'Zero-filter:      {n_valid}/{n_total} vertices kept')

        sub_labels   = parcel_labels[valid]
        sub_features = features[valid]

        mean = sub_features.mean(axis=0)
        std  = sub_features.std(axis=0)
        std[std < 1e-8] = 1.0   
        sub_features = ((sub_features - mean) / std).astype(np.float32)
        log.info(f'Z-score done')

        verts_per_region = np.array([np.sum(sub_labels == r) for r in unique_regions])
        log.info(f'Vertices/region:  min={verts_per_region.min()}, max={verts_per_region.max()}')

        mind_mat = compute_mind_matrix(
            sub_features, sub_labels, unique_regions,
            k=K_NN, min_verts=MIN_VERTS, logger=log)

        n_nan   = int(np.isnan(mind_mat).sum())
        od_mask = ~np.eye(N_REGIONS, dtype=bool)
        od_vals = mind_mat[od_mask]
        od_vals = od_vals[~np.isnan(od_vals)]

        log.info(f'QC:  shape={mind_mat.shape}  NaN_edges={n_nan}')

        assert mind_mat.shape == (N_REGIONS, N_REGIONS), f'Wrong shape: {mind_mat.shape}'
        assert np.all(np.diag(mind_mat) == 1.0), 'Diagonal != 1'
        assert np.allclose(mind_mat, mind_mat.T, equal_nan=True), 'Matrix not symmetric'
        assert float(od_vals.min()) >= 0.0, f'Negative MIND values: {od_vals.min()}'
        assert float(od_vals.max()) <= 1.0, f'MIND values > 1: {od_vals.max()}'

        np.save(str(out_npy), mind_mat)
        pd.DataFrame(mind_mat, index=region_names, columns=region_names).to_csv(str(out_csv))

        elapsed = time.time() - t0
        log.info(f'Saved -> {out_npy.name}')
        return subj_id, 'OK', elapsed, n_nan

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f'FAILED: {exc}\n{traceback.format_exc()}')
        return subj_id, f'FAILED: {exc}', elapsed, -1


def main():
    parser = argparse.ArgumentParser(description='HCP-S1200 MIND pipeline')
    parser.add_argument('--dry-run',  action='store_true')
    parser.add_argument('--subject', default=None)
    args = parser.parse_args()

    try:
        df = pd.read_excel(str(SUBJ_TABLE))
    except Exception:
        df = pd.read_csv(str(SUBJ_TABLE))
    
    if 'TwinPairID' in df.columns:
        df = df[df['TwinPairID'] != 'Pair41']
        
    id_col = next(c for c in df.columns if any(k in c.lower() for k in ('subject', 'id', 'subj')))
    df = df.dropna(subset=[id_col])
    subjects = [str(int(float(s))) for s in df[id_col].values]
    
    print(f'Subject list: {len(subjects)} valid entries')

    print(f'\nLoading HCP-MMP atlas...')
    parcel_labels, lh_idx, rh_idx, unique_regions, region_names = load_atlas(ATLAS_PATH)
    print(f'  Parcels: {len(unique_regions)} (expected {N_REGIONS})')

    if args.subject:
        subjects = [args.subject]
    elif args.dry_run:
        subjects = subjects[:1]

    already_done = [s for s in subjects
                    if (OUT_DIR / f'{s}_MIND.npy').exists() and
                       (OUT_DIR / f'{s}_MIND.csv').exists()]
    to_run = [s for s in subjects if s not in already_done]
    print(f'\nTo run: {len(to_run)}   Already done: {len(already_done)}\n')

    results = []

    if len(to_run) <= 1:
        for s in to_run:
            r = process_subject(s, parcel_labels, lh_idx, rh_idx, unique_regions, region_names)
            results.append(r)
            print(f'  {r[0]}  {r[1]}  {r[2]:.1f}s  NaN_edges={r[3]}')
    else:
        n_total = len(to_run)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(process_subject, s, parcel_labels, lh_idx, rh_idx,
                            unique_regions, region_names): s
                for s in to_run
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                n_ok   = sum(1 for _, st, _, _ in results if st == 'OK')
                n_fail = sum(1 for _, st, _, _ in results if st.startswith('FAILED'))
                print(f'[{len(results):>3}/{n_total}] {r[0]:>8}  {r[1]:<12}  '
                      f'{r[2]:>6.1f}s  nan={r[3]:>4}  (ok={n_ok} fail={n_fail})')

    for s in already_done:
        results.append((s, 'SKIPPED', 0.0, 0))

    print('\n' + '-' * 60)
    print('BATCH AUDIT')
    ok_r   = [r for r in results if r[1] == 'OK']
    skip_r = [r for r in results if r[1] == 'SKIPPED']
    fail_r = [r for r in results if r[1].startswith('FAILED')]
    print(f'  OK: {len(ok_r)}  Skipped: {len(skip_r)}  Failed: {len(fail_r)}')

    print('\nValidating output matrices...')
    bad, nan_heavy = [], []
    for s in subjects:
        npy = OUT_DIR / f'{s}_MIND.npy'
        if npy.exists():
            m = np.load(str(npy))
            if m.shape != (N_REGIONS, N_REGIONS):
                bad.append((s, m.shape))
            n_nan = int(np.isnan(m).sum())
            if n_nan > 0:
                nan_heavy.append((s, n_nan))

    n_checked = sum(1 for s in subjects if (OUT_DIR / f'{s}_MIND.npy').exists())
    if bad:
        print(f'  BAD SHAPES: {bad}')
    else:
        print(f'  Shapes OK: all {n_checked} matrices are {N_REGIONS}x{N_REGIONS}')
    print('-' * 60)
    print(f'Outputs  -> {OUT_DIR}')
    print(f'Done.')


if __name__ == '__main__':
    main()
