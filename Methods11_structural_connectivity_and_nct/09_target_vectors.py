import os
"""
Q4 prep: build cortical network target vectors
Translates B1 network definitions (Glasser label IDs from
phase8networkmaskmaker.py) into 360-node target vectors aligned to the
controllability node ordering (node_names.csv).

CORTICAL ONLY: the 50 Tian subcortical members of each network cannot be
represented in the 360-node cortical controllability space (MIND is a
cortical morphometric measure). Targets are therefore the cortical
component of each network. JHU (white-matter) members are also excluded.

Outputs -> NCT_inputs/Controllability/targets/
  target_vectors.npy        (4, 360) float64, unit-normalised rows
  target_membership.csv     node x network binary membership
  target_overlap.csv        pairwise network overlap (Jaccard + shared nodes)
  target_report.txt         per-network cortical node count + missing members
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

XML_PATH  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "reference_atlas/HCP-MMP/HCP-Multi-Modal-Parcellation-1.0.xml"))
NCT_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
NODE_CSV  = NCT_DIR / 'node_names.csv'
OUT_DIR   = NCT_DIR / 'Controllability' / 'targets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_NODES = 360

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('targets')

GLASSER_NETWORKS = {
    'Reward': [
        90, 1090,  72, 1072,  88, 1088,  65, 1065,
        91, 1091,  92, 1092,  93, 1093, 166, 1166,
        164, 1164, 165, 1165,
        111, 1111, 112, 1112,
        61, 1061, 180, 1180,
    ],
    'Salience': [
        111, 1111, 112, 1112, 109, 1109,
        57, 1057,  59, 1059,  40, 1040,
        79, 1079,  80, 1080,  81, 1081,
        113, 1113, 115, 1115, 114, 1114,
    ],
    'DMN': [
        161, 1161, 162, 1162,  35, 1035,
        33, 1033,  34, 1034,   14, 1014,
        65, 1065,  72, 1072,   88, 1088,
        27, 1027,  30, 1030,
        150, 1150, 143, 1143,
        128, 1128, 129, 1129,
        126, 1126, 155, 1155, 127, 1127,
    ],
    'Olfactory': [
        110, 1110,
        118, 1118, 119, 1119,
        131, 1131, 172, 1172,
        93, 1093, 166, 1166, 92, 1092,
        164, 1164,
    ],
}
NETWORK_ORDER = ['Reward', 'Olfactory', 'Salience', 'DMN']


def parse_glasser_xml(xml_path):
    content = xml_path.read_text(encoding='ISO-8859-1')
    idx_to_name = {}
    for m in re.finditer(r'<label index="(\d+)"[^>]*>([^<]+)</label>', content):
        idx, name = int(m.group(1)), m.group(2).strip()
        idx_to_name[idx] = name
    return idx_to_name


def glasser_id_to_roi_name(gid, idx_to_name):
    """Convert Glasser label ID (e.g. 110 or 1110) to node_names ROI form."""
    base = idx_to_name.get(gid)
    if base is None:
        return None
    return f'{base}_ROI'


def main():
    log.info('Q4 prep: building cortical network target vectors')

    idx_to_name = parse_glasser_xml(XML_PATH)

    node_df    = pd.read_csv(str(NODE_CSV))
    node_names = node_df['Region'].tolist()
    name_to_pos = {n: i for i, n in enumerate(node_names)}

    membership = np.zeros((N_NODES, len(NETWORK_ORDER)), dtype=int)
    missing_report = {}

    for ni, net in enumerate(NETWORK_ORDER):
        found, missing = [], []
        for gid in GLASSER_NETWORKS[net]:
            roi = glasser_id_to_roi_name(gid, idx_to_name)
            if roi is None:
                missing.append(f'GID{gid}(no XML entry)')
                continue
            pos = name_to_pos.get(roi)
            if pos is None:
                missing.append(f'{roi}(not in node_names)')
                continue
            membership[pos, ni] = 1
            found.append(roi)
        missing_report[net] = missing
        log.info(f'  {net:<10}: {len(found):>2} cortical nodes mapped, {len(missing)} missing')

    targets = np.zeros((len(NETWORK_ORDER), N_NODES), dtype=np.float64)
    for ni, net in enumerate(NETWORK_ORDER):
        v = membership[:, ni].astype(np.float64)
        norm = np.linalg.norm(v)
        if norm > 0:
            targets[ni] = v / norm
        else:
            log.error(f'  {net}: zero-norm target -- no members mapped!')

    np.save(str(OUT_DIR / 'target_vectors.npy'), targets)

    mem_df = pd.DataFrame(membership, columns=NETWORK_ORDER)
    mem_df.insert(0, 'Region', node_names)
    mem_df.insert(0, 'Position', range(N_NODES))
    mem_df.to_csv(str(OUT_DIR / 'target_membership.csv'), index=False)

    overlap_rows = []
    for i in range(len(NETWORK_ORDER)):
        for j in range(i+1, len(NETWORK_ORDER)):
            a = membership[:, i].astype(bool)
            b = membership[:, j].astype(bool)
            shared = int((a & b).sum())
            union  = int((a | b).sum())
            jac    = shared / union if union > 0 else 0.0
            overlap_rows.append({
                'Network_A': NETWORK_ORDER[i],
                'Network_B': NETWORK_ORDER[j],
                'Shared_nodes': shared,
                'Union_nodes': union,
                'Jaccard': round(jac, 4),
                'Shared_regions': ', '.join(
                    node_names[k] for k in range(N_NODES) if a[k] and b[k]),
            })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(str(OUT_DIR / 'target_overlap.csv'), index=False)

    report = [
        'NCT Q4 TARGET VECTORS -- CORTICAL NETWORK DEFINITIONS',
        '=' * 58,
        f'Source: B1 phase8networkmaskmaker.py (Glasser cortical members only)',
        f'Subcortical (Tian) and WM (JHU) members excluded -- not in 360-node cortical NCT space.',
        '',
        'Cortical node counts per network:',
    ]
    for net in NETWORK_ORDER:
        ni = NETWORK_ORDER.index(net)
        n  = int(membership[:, ni].sum())
        members = [node_names[k] for k in range(N_NODES) if membership[k, ni]]
        report.append(f'  {net:<10}: {n} nodes')
        report.append(f'    {", ".join(members)}')
        if missing_report[net]:
            report.append(f'    EXCLUDED (subcortical/missing): {missing_report[net]}')
    report += ['', 'Pairwise overlap:']
    for _, r in overlap_df.iterrows():
        report.append(f'  {r.Network_A} n {r.Network_B}: {r.Shared_nodes} shared (Jaccard {r.Jaccard})')

    (OUT_DIR / 'target_report.txt').write_text('\n'.join(report))
    log.info('Done. Next: nct_control_energy.py')


if __name__ == '__main__':
    main()
