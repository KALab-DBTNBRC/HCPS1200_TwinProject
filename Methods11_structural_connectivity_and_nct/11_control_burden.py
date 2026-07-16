import os
"""
NCT Q5: Optimal control trajectory & regional burden
Which regions must supply the most control input to drive the brain
from rest into a target network state?

Theory:
  Optimal input  u*(t) = e^{A(T-t)} W_c(T)^{-1} x_T   (x0=0, B=I)
  Regional burden  Burden_j = integral 0..T u*_j(t)^2 dt

Eigenbasis form (A=VLambdaV', no matrix inversion):
  p_tilde_i = (V'x_T)_i * 2*lambda_i/(e^{2*lambda_i*T}-1)
  G_ik = (e^{(lambda_i+lambda_k)T}-1)/(lambda_i+lambda_k)         [limit T as lambda_i+lambda_k->0]
  Burden_j = sum_i sum_k V_ji V_jk p_tilde_i p_tilde_k G_ik

Built-in validation:  sum_j Burden_j == E_min(x_T)  (Q4 energy)  -- checked.

Outputs (per substrate x 4 targets x T={1,3,5}) -> control_burden/:
  {sub}_burden_{net}_T{T}.npy   (238, 360)  per-subject regional burden
  burden_group_maps.csv         group-mean burden per node/network/substrate
  burden_concentration.csv      participation ratio + in/out-target ratio
  burden_control_role.csv       cross-target mean burden per node (driver map)
  burden_aud_association.csv    AUD LME + discordant on summary burden metrics
  burden_vs_ac.csv              relationship between burden and AC (mechanistic)
  burden_validation.csv         Sum burden vs Q4 energy per subject
"""

import warnings, logging, time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

CTRL_DIR  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs/Controllability"))
NCT_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
MIND_NORM = NCT_DIR / 'MIND_norm'
SC_NORM   = NCT_DIR / 'SC_norm'
TARGETS   = CTRL_DIR / 'targets' / 'target_vectors.npy'
MEMBER    = CTRL_DIR / 'targets' / 'target_membership.csv'
ENERGY    = CTRL_DIR / 'control_energy'
BEH_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/network_roi_metrics_FINAL.csv"))
NODE_CSV  = NCT_DIR / 'node_names.csv'
OUT_DIR   = CTRL_DIR / 'control_burden'
OUT_DIR.mkdir(exist_ok=True)

N_NODES       = 360
T_PRIMARY     = 3
T_ALL         = [1, 3, 5]
EIG_EPS       = 1e-10
NETWORK_ORDER = ['Reward', 'Olfactory', 'Salience', 'DMN']
N_PERM        = 5000
ALPHA_FDR     = 0.05

COVARIATES = ['Age_in_Yrs','Gender_Label','SSAGA_TB_Still_Smoking',
              'SSAGA_Times_Used_Illicits','SSAGA_Mj_Times_Used',
              'FamHist_Combined_DrgAlc']
CAT_COVS   = ['Race','Ethnicity']

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('NCT_Q5')


def compute_burden(eigvals, eigvecs, x_T, T):
    """
    Regional control burden for all subjects, one target, one horizon.
      eigvals (N,M)  eigvecs (N,M,M)  x_T (M,)  -> burden (N,M)
    """
    N, M = eigvals.shape

    inv_coeff = np.where(np.abs(eigvals) > EIG_EPS,
                         2.0*eigvals/np.expm1(2.0*eigvals*T),
                         1.0/T)

    x_tilde = np.einsum('nmi,m->ni', eigvecs, x_T)
    p_tilde = x_tilde * inv_coeff

    WS = eigvals[:, :, None] + eigvals[:, None, :]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        G = np.where(np.abs(WS) > EIG_EPS, np.expm1(WS*T)/WS, T)

    Mmat = p_tilde[:, :, None] * p_tilde[:, None, :] * G

    temp   = np.einsum('nji,nik->njk', eigvecs, Mmat)
    burden = np.einsum('njk,njk->nj', temp, eigvecs)
    return burden


def load_stack(directory, tag, subject_ids):
    return np.stack([np.load(str(directory / f'{s}_{tag}_norm.npy')).astype(np.float64)
                     for s in subject_ids], axis=0)


def load_metadata():
    subj_idx = pd.read_csv(str(CTRL_DIR / 'subject_index.csv'))
    beh      = pd.read_csv(str(BEH_CSV))
    subj_idx['subj_id'] = subj_idx['subj_id'].astype(int)
    beh['Subject']      = beh['Subject'].astype(int)
    df = subj_idx.merge(
        beh[['Subject','TwinPairID','ZygosityGT1','Severity','Age_in_Yrs',
             'Gender','Race','Ethnicity','SSAGA_TB_Still_Smoking',
             'SSAGA_Times_Used_Illicits','SSAGA_Mj_Times_Used',
             'FamHist_Combined_DrgAlc']],
        left_on='subj_id', right_on='Subject', how='left')
    df['Gender_Label'] = pd.Categorical(
        df['Gender'].map({0:'Female',1:'Male'}), categories=['Female','Male'])
    for col in ['Race','Ethnicity']:
        rare = df[col].value_counts()[lambda s: s<5].index
        df[col] = df[col].replace(rare,'Other_Unknown').astype(str)
    return subj_idx, df


def _fdr(pvals):
    from statsmodels.stats.multitest import multipletests
    pv=pd.to_numeric(pvals,errors='coerce').values
    out=np.full(len(pv),np.nan); v=~np.isnan(pv)
    if v.sum()>0: out[v]=multipletests(pv[v],method='fdr_bh')[1]
    return out


def participation_ratio(burden_row):
    """PR = (sum b)^2 / sum b^2  -- effective # of regions sharing the burden."""
    s1 = burden_row.sum()
    s2 = (burden_row**2).sum()
    return (s1*s1)/s2 if s2 > 0 else np.nan


def lme_metric(df, col):
    base = COVARIATES + [f'C({c})' for c in CAT_COVS]
    d = df.dropna(subset=[col,'Severity','TwinPairID'])
    res = smf.mixedlm(f'{col} ~ Severity + '+' + '.join(base), d,
                      groups=d['TwinPairID']).fit(method='cg', disp=False)
    return float(res.params.get('Severity',np.nan)), float(res.pvalues.get('Severity',np.nan))


def disc_metric(df, col):
    mz = df[df['ZygosityGT1'].str.startswith('MZ')]
    mz = mz.groupby('TwinPairID').filter(lambda g: len(g)==2)
    mz = mz.sort_values(['TwinPairID','Severity'],ascending=[True,False])
    dsev,dM=[],[]
    for _,g in mz.groupby('TwinPairID'):
        g=g.reset_index(drop=True)
        dsev.append(g.iloc[0]['Severity']-g.iloc[1]['Severity'])
        dM.append(g.iloc[0][col]-g.iloc[1][col])
    dsev,dM=np.array(dsev,float),np.array(dM,float)
    valid=~np.isnan(dM)&~np.isnan(dsev)
    if valid.sum()<5: return np.nan,np.nan
    rho,_=stats.spearmanr(dsev[valid],dM[valid])
    rng=np.random.default_rng(42)
    null=np.array([stats.spearmanr(dsev[valid],
                   dM[valid]*rng.choice([-1,1],valid.sum()))[0] for _ in range(N_PERM)])
    return round(float(rho),4), round(float((np.abs(null)>=abs(rho)).mean()),4)


def main():
    t0=time.time()
    log.info('NCT Q5: Optimal Control -- Regional Burden')

    subj_idx, df = load_metadata()
    subject_ids  = subj_idx['subj_id'].astype(str).tolist()
    N=len(subject_ids)
    targets   = np.load(str(TARGETS))
    node_names= pd.read_csv(str(NODE_CSV))['Region'].tolist()
    member    = pd.read_csv(str(MEMBER))

    log.info('Eigendecomposing MIND...')
    MIND=load_stack(MIND_NORM,'MIND',subject_ids)
    mind_w,mind_v=np.linalg.eigh(MIND); del MIND
    log.info('Eigendecomposing SC...')
    SC=load_stack(SC_NORM,'SC',subject_ids)
    sc_w,sc_v=np.linalg.eigh(SC); del SC

    group_rows=[]; concentration_rows=[]; validation_rows=[]
    burden_store={}

    for subname,(w,v) in [('MIND',(mind_w,mind_v)),('SC',(sc_w,sc_v))]:
        for k,net in enumerate(NETWORK_ORDER):
            x_T=targets[k]
            for T in T_ALL:
                burden=compute_burden(w,v,x_T,T)
                np.save(str(OUT_DIR/f'{subname}_burden_{net}_T{T}.npy'),
                        burden.astype(np.float32))

                if T==T_PRIMARY:
                    burden_store[(subname,net)]=burden
                    gm=burden.mean(0)
                    for j,nm in enumerate(node_names):
                        group_rows.append({'Substrate':subname,'Network':net,
                                           'Region':nm,'Burden_mean':round(float(gm[j]),8),
                                           'Burden_sd':round(float(burden[:,j].std()),8)})
                    pr=np.array([participation_ratio(burden[i]) for i in range(N)])
                    in_mask=member[net].values.astype(bool)
                    in_b =burden[:,in_mask].sum(1)
                    out_b=burden[:,~in_mask].sum(1)
                    concentration_rows.append({
                        'Substrate':subname,'Network':net,
                        'PR_mean':round(float(pr.mean()),2),
                        'PR_sd':round(float(pr.std()),2),
                        'InTarget_frac_mean':round(float((in_b/(in_b+out_b)).mean()),4),
                        'N_target_nodes':int(in_mask.sum())})
                    e_file=ENERGY/f'{subname}_network_energy_T{T}.npy'
                    if e_file.exists():
                        e=np.load(str(e_file))[:,k]
                        if subname=='SC': e=np.exp(e)
                        sb=burden.sum(1)
                        validation_rows.append({
                            'Substrate':subname,'Network':net,
                            'SumBurden_mean':round(float(sb.mean()),6),
                            'Q4Energy_mean':round(float(e.mean()),6),
                            'Max_abs_diff':round(float(np.abs(sb-e).max()),8)})

    pd.DataFrame(group_rows).to_csv(str(OUT_DIR/'burden_group_maps.csv'),index=False)
    pd.DataFrame(concentration_rows).to_csv(str(OUT_DIR/'burden_concentration.csv'),index=False)
    pd.DataFrame(validation_rows).to_csv(str(OUT_DIR/'burden_validation.csv'),index=False)

    for r in validation_rows:
        flag='OK' if r['Max_abs_diff']<1e-3 else 'MISMATCH'
        log.info(f"  {r['Substrate']} {r['Network']:<10}: Sum burden={r['SumBurden_mean']:.5f}  "
                 f"E={r['Q4Energy_mean']:.5f}  maxdiff={r['Max_abs_diff']:.2e}  {flag}")

    role_rows=[]
    for subname in ['MIND','SC']:
        stacked=np.stack([burden_store[(subname,net)].mean(0)
                          for net in NETWORK_ORDER],axis=1)
        role=stacked.mean(1)
        order=np.argsort(role)[::-1]
        for j in range(N_NODES):
            role_rows.append({'Substrate':subname,'Region':node_names[j],
                              'Mean_burden_across_targets':round(float(role[j]),8),
                              'Rank':int(np.where(order==j)[0][0])+1})
    pd.DataFrame(role_rows).to_csv(str(OUT_DIR/'burden_control_role.csv'),index=False)

    aud_rows=[]
    for subname in ['MIND','SC']:
        for net in NETWORK_ORDER:
            burden=burden_store[(subname,net)]
            in_mask=member[net].values.astype(bool)
            tmp=df.copy()
            tmp['_PR']=[participation_ratio(burden[i]) for i in range(N)]
            tmp['_inFrac']=burden[:,in_mask].sum(1)/burden.sum(1)
            for metric_col,label in [('_PR','ParticipationRatio'),
                                     ('_inFrac','InTargetFraction')]:
                b,p=lme_metric(tmp,metric_col)
                rho,pp=disc_metric(tmp,metric_col)
                aud_rows.append({'Substrate':subname,'Network':net,'Metric':label,
                                 'Sev_Beta':round(b,6),'Sev_P':round(p,4),
                                 'MZ_rho':rho,'MZ_perm_p':pp})
    aud_df=pd.DataFrame(aud_rows)
    aud_df['Sev_FDR']=_fdr(aud_df['Sev_P'])
    aud_df['MZ_FDR'] =_fdr(aud_df['MZ_perm_p'])
    aud_df.to_csv(str(OUT_DIR/'burden_aud_association.csv'),index=False)

    vac_rows=[]
    for subname,acf in [('MIND','MIND_AC_T3.npy'),('SC','SC_AC_T3.npy')]:
        ac=np.load(str(CTRL_DIR/acf)).mean(0)
        if subname=='SC': ac=np.log(ac)
        for net in NETWORK_ORDER:
            role=burden_store[(subname,net)].mean(0)
            r,_=stats.spearmanr(ac,role)
            vac_rows.append({'Substrate':subname,'Network':net,
                             'Spearman_AC_vs_burden':round(float(r),4)})
    pd.DataFrame(vac_rows).to_csv(str(OUT_DIR/'burden_vs_ac.csv'),index=False)

    log.info(f'Q5 COMPLETE   Wall: {timedelta(seconds=int(time.time()-t0))}')
    log.info(f'Outputs -> {OUT_DIR}/')
    log.info('All five NCT questions complete.')


if __name__=='__main__':
    main()
