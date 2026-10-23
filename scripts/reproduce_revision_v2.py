"""Revision v2 analysis (21/10/2026): one loss implementation for all main-table rows, Holm-adjusted RQ1
contrasts, no-pretraining control, recipe sensitivity (RQ2) and external evaluation (RQ3).

Inputs are read-only: results/utah2018_run2_master (E2, E3, E4, E7 unchanged), output/analysis (E7, nnU-Net),
results/revision_v2_20261021 (FP32 reruns of E0/E1/E5/E6 and all revision runs).

Run: python -B scripts/analyze_revision_v2_20261021.py --paper-dir <overleaf working copy> [--figures]
Writes output/analysis/revision_v2_20261021.json and, in --paper-dir, paired_effects.tex,
revision_analysis_20261006.tex (figure environments), recipe_external_tables.tex and the three figures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results'
ANALYSIS = ROOT / 'results'
REV = ROOT / 'results/revision_v2'
SEEDS = (42, 2026, 3407)
EXCLUDE = 'CMPXO4J23G58J53Q98SZ'
JSON_PATH = ROOT / 'outputs/revision_v2.json'
FIGURES = ['forest_revision_20261006', 'case_differences_revision_20261006', 'pareto_revision_20261006']

# Main-table configurations. E0/E1/E5/E6 now come from the FP32-loss reruns, so every row shares one loss path.
MAIN = {
    'E0': REV / 'fp32_loss/e0_seed{s}', 'E1': REV / 'fp32_loss/e1_seed{s}',
    'E2': BASE / 'E2_Large_Full_FT/seed{s}', 'E3': BASE / 'E3_Large_Partial_FT/seed{s}',
    'E4': BASE / 'E4_Giant_LoRA/seed{s}', 'E5': REV / 'fp32_loss/e5_seed{s}',
    'E6': REV / 'fp32_loss/e6_seed{s}', 'E7': ANALYSIS / 'E7_Giant_Partial_FT/seed{s}',
}
EXTRA = {  # revision runs (three seeds unless stated)
    'RAND': REV / 'no_pretraining/e2rand_seed{s}', 'RAND_TUNED': REV / 'no_pretraining_tuned/e2rand_tuned_seed{s}',
    'LORA_L_1e-4': REV / 'lora_lr1e-4/e6_seed{s}', 'LORA_G_1e-4': REV / 'lora_lr1e-4/e4_seed{s}',
    'E2_CLIP': REV / 'fullft_clip/e2_seed{s}',
    'F0': REV / 'readout/F0/seed{s}', 'F1': REV / 'readout/F1/seed{s}',
    'L0': REV / 'readout/L0/seed{s}', 'L1': REV / 'readout/L1/seed{s}',
    # Common improved protocol (post hoc): four-layer readout + clipping 1.0 + learning-rate rule, ViT-Large, RTX 5090.
    'V2_FR': REV / 'protocol_v2/V_FR/seed{s}', 'V2_PA': REV / 'protocol_v2/V_PA/seed{s}',
    'V2_LO': REV / 'protocol_v2/V_LO/seed{s}', 'V2_FU': REV / 'protocol_v2/V_FU/seed{s}',
}
SINGLE = {'E2_NOCLIP_R': REV / 'fullft_noclip_5090/e2_noclip_seed42'}  # one seed (42): unclipped Full FT retrained on a second GPU
NAMES = {'E0': 'Large Frozen (BCE)', 'E1': 'Large Frozen', 'E2': 'Large Full FT', 'E3': 'Large Partial FT',
         'E4': 'Giant LoRA', 'E5': 'Giant Frozen', 'E6': 'Large LoRA', 'E7': 'Giant Partial FT',
         'nnU-Net': 'nnU-Net ResEncM 3D'}
# RQ1 family (Holm, 16 contrasts) = the contrasts reported in tab:paired.
CONTRASTS = {
    'E2 - E4': {'E2': 1, 'E4': -1}, 'E2 - E1': {'E2': 1, 'E1': -1}, 'E3 - E1': {'E3': 1, 'E1': -1},
    'E6 - E1': {'E6': 1, 'E1': -1}, 'E4 - E5': {'E4': 1, 'E5': -1},
    'E5 - E1 (Frozen scale)': {'E5': 1, 'E1': -1}, 'E4 - E6 (LoRA scale)': {'E4': 1, 'E6': -1},
    'LoRA - Frozen scale interaction': {'E4': 1, 'E6': -1, 'E5': -1, 'E1': 1},
    'E7 - E5': {'E7': 1, 'E5': -1}, 'E4 - E7': {'E4': 1, 'E7': -1},
    'E7 - E3 (Partial scale)': {'E7': 1, 'E3': -1},
    'Partial - Frozen scale interaction': {'E7': 1, 'E3': -1, 'E5': -1, 'E1': 1},
    'E6 - E7': {'E6': 1, 'E7': -1},
    'nnU-Net - E2': {'nnU-Net': 1, 'E2': -1}, 'nnU-Net - E4': {'nnU-Net': 1, 'E4': -1},
    'nnU-Net - E7': {'nnU-Net': 1, 'E7': -1},
}
OTHER_CONTRASTS = {  # reported with unadjusted CIs, labelled exploratory
    'E0 - E1 (objective ablation)': {'E0': 1, 'E1': -1},
    'E2 - RAND (pretraining, full FT)': {'E2': 1, 'RAND': -1},
    'E1 - RAND (frozen DINOv2 vs scratch)': {'E1': 1, 'RAND': -1},
    'RAND_TUNED - RAND (scratch recipe)': {'RAND_TUNED': 1, 'RAND': -1},
    'E2 - RAND_TUNED (pretraining vs tuned scratch)': {'E2': 1, 'RAND_TUNED': -1},
    'LoRA LR, Large (1e-4 - 1e-5)': {'LORA_L_1e-4': 1, 'E6': -1},
    'LoRA LR, Giant (1e-4 - 1e-5)': {'LORA_G_1e-4': 1, 'E4': -1},
    'Giant - Large LoRA at 1e-4': {'LORA_G_1e-4': 1, 'LORA_L_1e-4': -1},
    'Scale-gain change (1e-4 - 1e-5)': {'LORA_G_1e-4': 1, 'LORA_L_1e-4': -1, 'E4': -1, 'E6': 1},
    'E2 - LoRA Large at 1e-4': {'E2': 1, 'LORA_L_1e-4': -1},
    'E2 - LoRA Giant at 1e-4': {'E2': 1, 'LORA_G_1e-4': -1},
    'Clipping (Full FT, clip - no clip)': {'E2_CLIP': 1, 'E2': -1},
    'Clipped Full FT - LoRA Large at 1e-4': {'E2_CLIP': 1, 'LORA_L_1e-4': -1},
    'Clipped Full FT - LoRA Giant at 1e-4': {'E2_CLIP': 1, 'LORA_G_1e-4': -1},
    'E2 - E6 (Large Full FT - Large LoRA)': {'E2': 1, 'E6': -1},
    'Readout, Frozen (four - last)': {'F1': 1, 'F0': -1},
    'Readout, LoRA (four - last)': {'L1': 1, 'L0': -1},
    'V2 - V1, Frozen': {'V2_FR': 1, 'E1': -1}, 'V2 - V1, Partial FT': {'V2_PA': 1, 'E3': -1},
    'V2 - V1, LoRA': {'V2_LO': 1, 'E6': -1}, 'V2 - V1, Full FT': {'V2_FU': 1, 'E2': -1},
    'V2: Full FT - LoRA': {'V2_FU': 1, 'V2_LO': -1}, 'V2: LoRA - Partial FT': {'V2_LO': 1, 'V2_PA': -1},
    'V2: Partial FT - Frozen': {'V2_PA': 1, 'V2_FR': -1}, 'V2: Full FT - Frozen': {'V2_FU': 1, 'V2_FR': -1},
    'V2: LoRA - Frozen': {'V2_LO': 1, 'V2_FR': -1},
}
TABLE_GROUPS = [
    ('A. Adaptation versus frozen extraction, same backbone', [
        ('E2 - E1', 'Large: Full FT $-$ Frozen'), ('E3 - E1', 'Large: Partial FT $-$ Frozen'),
        ('E6 - E1', 'Large: LoRA $-$ Frozen'), ('E4 - E5', 'Giant: LoRA $-$ Frozen'),
        ('E7 - E5', 'Giant: Partial FT $-$ Frozen')]),
    ('B. Backbone scale, Giant $-$ Large, same strategy', [
        ('E5 - E1 (Frozen scale)', 'Frozen'), ('E4 - E6 (LoRA scale)', 'LoRA'),
        ('E7 - E3 (Partial scale)', 'Partial FT (last four blocks)')]),
    ('C. Scale interaction: difference between scale gains', [
        ('LoRA - Frozen scale interaction', 'LoRA $-$ Frozen'),
        ('Partial - Frozen scale interaction', 'Partial FT $-$ Frozen')]),
    ('D. Practical configuration comparisons', [
        ('E2 - E4', 'Large Full FT $-$ Giant LoRA'), ('E4 - E7', 'Giant LoRA $-$ Giant Partial FT'),
        ('E6 - E7', 'Large LoRA $-$ Giant Partial FT')]),
    ('E. Task-specific reference with a distinct recipe', [
        ('nnU-Net - E2', 'nnU-Net $-$ Large Full FT'), ('nnU-Net - E4', 'nnU-Net $-$ Giant LoRA'),
        ('nnU-Net - E7', 'nnU-Net $-$ Giant Partial FT')]),
]
CODES = {name: name.split(' (')[0].replace(' - ', r' $-$ ') for name in CONTRASTS}
EXT_CONFIGS = ['E0', 'E1', 'E5', 'E3', 'E7', 'E6', 'E4', 'E2']  # external table order (Large/Giant pairs)
EXT_FP32 = {'E0', 'E1', 'E5', 'E6'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(d):
    assert d.ndim == 1 and np.isfinite(d).all()
    rng = np.random.default_rng(20260926)  # reset for each contrast and subset
    draws = d[rng.integers(0, len(d), (20000, len(d)))].mean(axis=1)
    p = min(1.0, 2*min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return {'difference_pp': float(100*d.mean()), 'ci95_pp': (100*np.quantile(draws, [.025, .975])).tolist(),
            'p_two_sided_bootstrap': p, 'median_pp': float(100*np.median(d)),
            'positive_cases': int((d > 0).sum()), 'n_cases': len(d)}


def cluster_bootstrap(d, groups):
    """Resample whole clusters (CDEMRIS patients with pre- and post-ablation volumes); same RNG rule."""
    assert d.ndim == 1 and np.isfinite(d).all() and len(groups) == len(d)
    labels = sorted(set(groups))
    sums = np.array([d[[g == lab for g in groups]].sum() for lab in labels])
    sizes = np.array([sum(g == lab for g in groups) for lab in labels])
    rng = np.random.default_rng(20260926)
    idx = rng.integers(0, len(labels), (20000, len(labels)))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    p = min(1.0, 2*min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return {'difference_pp': float(100*d.mean()), 'ci95_pp': (100*np.quantile(draws, [.025, .975])).tolist(),
            'p_two_sided_bootstrap': p, 'median_pp': float(100*np.median(d)),
            'positive_cases': int((d > 0).sum()), 'n_cases': len(d), 'n_clusters': len(labels),
            'resampling': 'patients (pre/post volumes kept together)'}


def holm(pvalues):
    names = sorted(pvalues, key=pvalues.get)
    m, out, running = len(names), {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i)*pvalues[name]))
        out[name] = running
    return out


def analyze():
    sources = {}

    def track(path):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return path

    ids = sorted(track(ROOT / 'splits/test_patients.txt').read_text().split())
    assert len(ids) == len(set(ids)) == 31 and EXCLUDE in ids

    def read_case_csv(path, seed):
        with track(path).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        mapping = {r['patient_id']: r for r in rows}
        assert len(rows) == 31 and sorted(mapping) == ids, path
        assert all(int(r['seed']) == seed for r in rows), path
        dice = np.array([float(mapping[i]['dice_3d']) for i in ids])
        iou = np.array([float(mapping[i]['iou_3d']) for i in ids])
        assert np.isfinite(dice).all() and ((0 <= dice) & (dice <= 1)).all()
        return dice, iou

    dice, iou = {}, {}
    for key, pattern in {**MAIN, **EXTRA}.items():
        pairs = [read_case_csv(Path(str(pattern).format(s=s)) / 'test_patient_metrics.csv', s) for s in SEEDS]
        dice[key] = np.array([p[0] for p in pairs])
        iou[key] = np.array([p[1] for p in pairs])
    for key, folder in SINGLE.items():
        d1, i1 = read_case_csv(folder / 'test_patient_metrics.csv', 42)
        dice[key], iou[key] = d1[None, :], i1[None, :]
    nn_d, nn_i = [], []
    for seed in SEEDS:
        path = ANALYSIS / f'nnUNet_ResEncM_3D/seed{seed}/test_case_metrics.json'
        rows = {r['case_id']: r for r in json.loads(track(path).read_text(encoding='utf-8'))}
        assert sorted(rows) == ids
        nn_d.append([float(rows[i]['dice_3d']) for i in ids])
        nn_i.append([float(rows[i].get('iou_3d', np.nan)) for i in ids])
    dice['nnU-Net'], iou['nnU-Net'] = np.array(nn_d), np.array(nn_i)

    keep30 = np.array([i != EXCLUDE for i in ids])
    summaries = {}
    for key in dice:
        summaries[key] = {}
        for subset, keep in [('all31', np.ones(31, dtype=bool)), ('exclude30', keep30)]:
            sm = dice[key][:, keep].mean(axis=1)
            im = iou[key][:, keep].mean(axis=1)
            summaries[key][subset] = {
                'mean': float(sm.mean()), 'sample_sd': float(sm.std(ddof=1)) if len(sm) > 1 else None,
                'seed_means': sm.tolist(), 'iou_mean': float(np.nanmean(im)),
                'iou_sample_sd': float(np.std(im, ddof=1)) if len(im) > 1 and np.isfinite(im).all() else None,
                'case_median': float(np.median(dice[key][:, keep].mean(axis=0)))}

    def contrast(weights):
        d = sum(w*dice[k] for k, w in weights.items()).mean(axis=0)
        per_seed = (100*sum(w*dice[k] for k, w in weights.items()).mean(axis=1)).tolist()
        return {'all31': bootstrap(d), 'exclude30': bootstrap(d[keep30]), 'seed_mean_differences_pp': per_seed,
                'by_case_pp': dict(zip(ids, (100*d).tolist()))}

    effects = {name: contrast(w) for name, w in CONTRASTS.items()}
    adjusted = holm({n: e['all31']['p_two_sided_bootstrap'] for n, e in effects.items()})
    for name in effects:
        effects[name]['all31']['p_holm_16'] = adjusted[name]
    other = {name: contrast(w) for name, w in OTHER_CONTRASTS.items()}
    external = analyze_external(track)
    return {'bootstrap': {'resamples': 20000, 'seed': 20260926, 'reset_rng_each_contrast_subset': True,
                          'ci': 'percentile, numpy default linear quantiles',
                          'p_value': 'two-sided, 2*min(P(draw<=0), P(draw>=0)) from the same bootstrap draws',
                          'holm_family': list(CONTRASTS)},
            'case_ids': ids, 'excluded_case': EXCLUDE, 'summaries': summaries,
            'rq1_paired_effects': effects, 'other_effects': other, 'external': external,
            'source_files': [{'path': k, 'sha256': v} for k, v in sorted(sources.items())],
            'generator_sha256': digest(Path(__file__)), 'numpy_version': np.__version__}


def analyze_external(track):
    """CDEMRIS (40 LGE, two centres) and MSD Task02 (20 bSSFP); inference only."""
    def load(folder):
        out = {}
        for seed in SEEDS:
            with track(folder / f'seed{seed}.csv').open(encoding='utf-8-sig', newline='') as stream:
                for r in csv.DictReader(stream):
                    out.setdefault((r['dataset'], r['case']), {})[seed] = float(r['dice_3d'])
        return out
    data = {k: load(REV / ('external/dino_fp32' if k in EXT_FP32 else 'external/dino_original') / k)
            for k in EXT_CONFIGS}
    nn = {}
    for seed in SEEDS:
        with track(REV / f'external/nnunet/seed{seed}.csv').open(encoding='utf-8-sig', newline='') as stream:
            for r in csv.DictReader(stream):
                nn.setdefault((r['dataset'], r['case']), {})[seed] = float(r['dice_3d'])
    data['nnU-Net'] = nn
    subsets = {'cdemris40': lambda k: k[0].startswith('cdemris'),
               'cdemris39': lambda k: k[0].startswith('cdemris') and k[1] != 'post_p18',
               'msd20': lambda k: k[0] == 'msd'}
    result = {}
    for key, cases in data.items():
        result[key] = {}
        for name, keep in subsets.items():
            ks = sorted(c for c in cases if keep(c))
            seed_means = [float(np.mean([cases[c][s] for c in ks])) for s in SEEDS]
            entry = {'n_cases': len(ks), 'mean': float(np.mean(seed_means)),
                     'sample_sd': float(np.std(seed_means, ddof=1)), 'seed_means': seed_means}
            if key != 'nnU-Net':
                d = np.array([np.mean([cases[c][s] for s in SEEDS]) - np.mean([nn[c][s] for s in SEEDS]) for c in ks])
                if name.startswith('cdemris'):
                    # pre_pXX and post_pXX are the same CDEMRIS patient; resample patients, not volumes.
                    entry['minus_nnunet'] = cluster_bootstrap(d, [c[1].split('_', 1)[1] for c in ks])
                else:
                    entry['minus_nnunet'] = bootstrap(d)
            result[key][name] = entry
    return result


def num(v):
    """Two decimals, or three when two would print a signed zero at a CI boundary."""
    if abs(v) < 0.0005:
        return '0.000'
    return f'{v:.3f}' if abs(v) < 0.005 else f'{v:.2f}'


def fmt_cell(x, p=None):
    return f"${num(x['difference_pp'])}\\;[{num(x['ci95_pp'][0])},\\,{num(x['ci95_pp'][1])}]$"


def fmt_p(p):
    return '$<0.001$' if p < 0.001 else f'${p:.3f}$'


def paired_table(report):
    lines = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Paired Dice contrasts (percentage points) for the research questions of the main protocol. '
             r'For a contrast A $-$ B, positive values indicate higher mean Dice for A; interactions are differences '
             r'between two Giant $-$ Large scale gains. Seeds are averaged within case. 95\% percentile case-bootstrap '
             r'CIs use 20,000 resamples (RNG seed 20260926, reset for each contrast and subset); $p_{\mathrm{Holm}}$ '
             r'adjusts two-sided bootstrap $p$ values over the 16 contrasts shown. Per-seed differences are given for '
             r'every contrast because the case bootstrap does not include retraining variance. Intervals are '
             r'conditional on the observed runs and assume exchangeable test volumes; patient independence is '
             r'unverified. The 30-case column excludes the identified near-duplicate without retraining.}',
             r'\label{tab:paired}', r'\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}',
             r'\toprule Comparison & 31 cases: $\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ & Per seed & '
             r'30 cases: $\Delta$ [95\% CI]\\ \midrule']
    for index, (title, rows) in enumerate(TABLE_GROUPS):
        if index:
            lines.append(r'\addlinespace[2pt]')
        lines.append(r'\multicolumn{5}{@{}l}{\textit{' + title + r'}}\\')
        for name, label in rows:
            e = report['rq1_paired_effects'][name]
            seeds = '/'.join(f'{v:.2f}' for v in e['seed_mean_differences_pp'])
            lines.append(r'\quad ' + label + ' & ' + fmt_cell(e['all31']) + ' & ' + fmt_p(e['all31']['p_holm_16'])
                         + ' & ' + seeds.replace('-', '$-$') + ' & ' + fmt_cell(e['exclude30']) + r'\\')
    lines += [r'\bottomrule\end{tabularx}', r'\par\smallskip\raggedright\scriptsize',
              'All DINOv2 rows use the BCE--Dice objective and one loss implementation. Partial FT updates the last '
              'four blocks, not a matched fraction of depth. nnU-Net uses its distinct 3D recipe. Per-seed values '
              'follow seed order 42/2026/3407.', r'\end{table}']
    return '\n'.join(lines) + '\n'


def protocol_v2_table(report):
    s, o = report['summaries'], report['other_effects']

    def pm(key):
        x = s[key]['all31']
        return f"\\pmv{{{x['mean']:.4f}}}{{{x['sample_sd']:.4f}}}"

    def cells(name):
        a = o[name]['all31']
        seeds = '/'.join(num(v) for v in o[name]['seed_mean_differences_pp'])
        return f"{fmt_cell(a)} & {a['positive_cases']}/31 & {seeds}"
    rows = [(k1, k2, label) for k1, k2, label in (('E1', 'V2_FR', 'Frozen'), ('E3', 'V2_PA', 'Partial FT'),
                                                  ('E6', 'V2_LO', 'LoRA'), ('E2', 'V2_FU', 'Full FT'))]
    lines = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Common improved protocol for the four ViT-Large strategies (post hoc; three seeds; all runs on one '
             r'RTX 5090 and software stack). Every strategy uses the four-layer readout, gradient-norm clipping at 1.0 and '
             r'the learning-rate rule (pretrained weights $10^{-5}$, LoRA adapters $10^{-4}$, decoder $10^{-3}$); all other '
             r'settings follow the main protocol, and the test set was evaluated only after all 12 runs had finished. '
             r'$\Delta$: paired case-bootstrap difference in percentage points (unadjusted 95\% CI); per-seed differences '
             r'follow seed order 42/2026/3407.}', r'\label{tab:v2}',
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xccrcr@{}}',
             r'\toprule Strategy (ViT-Large) & Main protocol & Common improved & $\Delta$ [95\% CI] & Cases $>0$ & Per seed\\ \midrule']
    for k1, k2, label in rows:
        lines.append(f"{label} & {pm(k1)} & {pm(k2)} & {cells('V2 - V1, ' + label)}\\\\")
    lines += [r'\midrule \multicolumn{6}{@{}l}{\textit{Contrasts under the common improved protocol}}\\']
    for name, label in (('V2: Full FT - LoRA', 'Full FT $-$ LoRA'), ('V2: LoRA - Partial FT', 'LoRA $-$ Partial FT'),
                        ('V2: Partial FT - Frozen', 'Partial FT $-$ Frozen'), ('V2: Full FT - Frozen', 'Full FT $-$ Frozen')):
        lines.append(f"\\quad {label} & & & {cells(name)}\\\\")
    lines += [r'\bottomrule\end{tabularx}', r'\end{table}', '']
    return '\n'.join(lines)


def recipe_external_tables(report):
    s, o, ext = report['summaries'], report['other_effects'], report['external']

    def pm(key, subset='all31'):
        x = s[key][subset]
        return f"\\pmv{{{x['mean']:.4f}}}{{{x['sample_sd']:.4f}}}"

    def ci(name):
        a = o[name]['all31']
        return (f"${num(a['difference_pp'])}\\;[{num(a['ci95_pp'][0])},\\,{num(a['ci95_pp'][1])}]$ & "
                f"{a['positive_cases']}/31 & " + '/'.join(f'{v:.2f}' for v in o[name]['seed_mean_differences_pp']).replace('-', '$-$'))
    rq2 = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
           r'\begin{table}[!htbp]', r'\centering\scriptsize',
           r'\caption{Recipe sensitivity within each strategy (RQ2; post hoc, three seeds, selection on validation '
           r'only). Each factor was varied for the strategy shown; rows are not compared across strategies. '
           r'$\Delta$ is the paired case-bootstrap difference (percentage points, unadjusted 95\% CI) against the '
           r'reference stated in the row; per-seed differences follow seed order 42/2026/3407.}',
           r'\label{tab:recipe}', r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xccrcr@{}}',
           r'\toprule Factor (strategy) & Reference & Variant & $\Delta$ [95\% CI] & Cases $>0$ & Per seed\\ \midrule',
           r'Adapter LR $10^{-4}$ (Large LoRA) & ' + pm('E6') + ' & ' + pm('LORA_L_1e-4') + ' & ' + ci('LoRA LR, Large (1e-4 - 1e-5)') + r'\\',
           r'Adapter LR $10^{-4}$ (Giant LoRA) & ' + pm('E4') + ' & ' + pm('LORA_G_1e-4') + ' & ' + ci('LoRA LR, Giant (1e-4 - 1e-5)') + r'\\',
           r'Giant $-$ Large at LR $10^{-4}$ (LoRA) & ' + pm('LORA_L_1e-4') + ' & ' + pm('LORA_G_1e-4') + ' & ' + ci('Giant - Large LoRA at 1e-4') + r'\\',
           r'Gradient clipping 1.0 (Large Full FT) & ' + pm('E2') + ' & ' + pm('E2_CLIP') + ' & ' + ci('Clipping (Full FT, clip - no clip)') + r'\\',
           r'Four-layer readout (Large Frozen) & ' + pm('F0') + ' & ' + pm('F1') + ' & ' + ci('Readout, Frozen (four - last)') + r'\\',
           r'Four-layer readout (Large LoRA) & ' + pm('L0') + ' & ' + pm('L1') + ' & ' + ci('Readout, LoRA (four - last)') + r'\\',
           r'\bottomrule\end{tabularx}', r'\par\smallskip\raggedright\scriptsize',
           r'The adapter learning rate was selected on validation in a single-seed sweep of Large LoRA ($10^{-5}$, '
           r'$10^{-4}$; $10^{-3}$ diverged) and applied unchanged to Giant LoRA. Clipped Full FT runs used an RTX 5090 '
           r'and the reference runs an RTX PRO 6000; Giant LoRA reference runs used an RTX 4000 Ada and its $10^{-4}$ runs an '
           r'RTX 4090. An unclipped seed-42 rerun on the RTX 5090 reached Dice '
           f"{s['E2_NOCLIP_R']['all31']['mean']:.4f}, against {s['E2_CLIP']['all31']['seed_means'][0]:.4f} with clipping "
           r'(its validation MDD is in Table~\ref{tab:stability}). Readout cells are independent retrainings with the main protocol '
           r'otherwise unchanged; seeds 42 and 2026 used an RTX 4090 and seed 3407 an RTX 5090, with the four cells of a '
           r'seed on one GPU. The four-layer readout partitions the original projection and adds no trainable parameters.',
           r'\end{table}', '']
    rows = []
    for k in EXT_CONFIGS + ['nnU-Net']:
        e = ext[k]

        def m(sub):
            return f"\\pmv{{{e[sub]['mean']:.3f}}}{{{e[sub]['sample_sd']:.3f}}}"

        def dlt(sub):
            if k == 'nnU-Net':
                return '--'
            a = e[sub]['minus_nnunet']
            return f"${a['difference_pp']:.1f}\\;[{a['ci95_pp'][0]:.1f},\\,{a['ci95_pp'][1]:.1f}]$"
        rows.append(f"{NAMES[k]} & {m('cdemris40')} & {dlt('cdemris40')} & {m('msd20')} & {dlt('msd20')}\\\\")
    rq3 = [r'\begin{table}[!htbp]', r'\centering\scriptsize',
           r'\caption{External evaluation without retraining (RQ3, exploratory). CDEMRIS: 40 LGE volumes, one '
           r'pre- and one post-ablation scan from each of 20 patients, two non-Utah centers; MSD Task02: 20 bSSFP '
           r'volumes (different sequence). Dice is the mean $\pm$ SD of three seed means; $\Delta$ is the paired '
           r'difference to nnU-Net in percentage points with an unadjusted 95\% bootstrap CI that resamples patients '
           r'(CDEMRIS) or volumes (MSD). Label definitions across centers may differ.}',
           r'\label{tab:external}', r'\begin{tabularx}{\linewidth}{@{}Xcccc@{}}',
           r'\toprule Configuration & CDEMRIS Dice & $\Delta$ vs nnU-Net & MSD Dice & $\Delta$ vs nnU-Net\\ \midrule']
    rq3 += rows + [r'\bottomrule\end{tabularx}', r'\end{table}']
    return '\n'.join(rq2 + rq3) + '\n'


SHORT = {'E1': 'L Frozen', 'E2': 'L Full FT*', 'E3': 'L Partial', 'E4': 'G LoRA', 'E5': 'G Frozen', 'E6': 'L LoRA',
         'E7': 'G Partial'}


def figure_envs():
    captions = [
        ('fig:revision-forest', 'Paired Dice contrasts of Table~\\ref{tab:paired}, in the same groups, for all 31 '
         'cases (circles) and the 30-case exclusion sensitivity (squares). Lines denote 95\\% percentile '
         'case-bootstrap CIs; the vertical reference is zero. These intervals do not include retraining variance '
         'or certify patient independence.'),
        ('fig:revision-cases', 'Empirical cumulative distributions of seed-averaged paired Dice differences for all '
         '31 cases. Every marker is one case; the outlined marker identifies the flagged near-duplicate. Positive '
         'values favor the first configuration.'),
        ('fig:revision-pareto', 'Observed trade-offs between accuracy and training resources under the main '
         'protocol. Dice is the three-seed mean with sample SD bars; resources come from the fixed workload (batch 4, '
         '448 by 448, RTX 4090, three repeats of 100 batches). Dashed lines mark the observed nondominated points in '
         'each panel. L and G denote ViT-Large and ViT-Giant; FT, fine-tuning. Large Full FT (*) had one skipped '
         'optimizer update during the benchmark.')]
    files = []
    for group in (zip(FIGURES[:2], captions[:2]), zip(FIGURES[2:], captions[2:])):
        tex = [r'% Generated by scripts/analyze_revision_v2_20261021.py.']
        for filename, (label, caption) in group:
            tex += [r'\begin{figure}[!htbp]', r'\centering', r'\includegraphics[width=\linewidth]{'+filename+'.pdf}',
                    r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{figure}']
        files.append('\n'.join(tex) + '\n')
    return files


def plot(report, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.savefig(outdir / f'{name}.pdf', bbox_inches='tight', metadata={'Creator': 'analyze_revision_v2_20261021.py'})
        plt.close(fig)

    eff = report['rq1_paired_effects']
    fig, ax = plt.subplots(figsize=(9, 8))
    names, positions, ticks, labels, y = [], [], [], [], 0.0
    for title, rows in TABLE_GROUPS:
        ticks.append(y); labels.append(title.replace('$-$', '−')); y += 1
        for name, label in rows:
            names.append(name); positions.append(y); ticks.append(y); labels.append(label.replace('$-$', '−')); y += 1
        y += .4
    positions = np.array(positions)
    for offset, subset, marker, color, lab in [(-.13, 'all31', 'o', '#17669b', 'All 31 cases'),
                                               (.13, 'exclude30', 's', '#c27619', 'Exclude flagged case (30)')]:
        mid = np.array([eff[n][subset]['difference_pp'] for n in names])
        ci = np.array([eff[n][subset]['ci95_pp'] for n in names])
        ax.errorbar(mid, positions+offset, xerr=[mid-ci[:, 0], ci[:, 1]-mid], fmt=marker, color=color,
                    markersize=4, capsize=2, label=lab)
    ax.set_yticks(ticks, labels)
    heads = {t.replace('$-$', '−') for t, _ in TABLE_GROUPS}
    for tick, text in zip(ax.get_yticklabels(), labels):
        if text in heads:
            tick.set_fontweight('bold')
    ax.tick_params(axis='y', length=0); ax.invert_yaxis(); ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel('Paired Dice difference (percentage points)'); ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); save(fig, FIGURES[0])

    lab = {n: l.replace('$-$', '−') for _, rows in TABLE_GROUPS for n, l in rows}
    ids = report['case_ids']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, name in zip(axes, ['E6 - E7', 'nnU-Net - E2', 'nnU-Net - E4']):
        d = eff[name]['by_case_pp']; ordered = sorted(d, key=d.get); x = [d[i] for i in ordered]
        yy = np.arange(1, 32)/31
        ax.step(x, yy, where='post', color='#17669b', lw=1); ax.scatter(x, yy, s=12, color='#17669b')
        j = ordered.index(EXCLUDE)
        ax.scatter([x[j]], [yy[j]], s=65, facecolors='none', edgecolors='#c27619', linewidths=1.5)
        ax.axvline(0, color='gray', lw=1, ls=':')
        ax.set(title=lab[name], xlabel='Dice difference (percentage points)', ylim=(0, 1.04)); ax.grid(alpha=.2)
    axes[0].set_ylabel('Cumulative fraction of 31 cases')
    fig.tight_layout(); save(fig, FIGURES[1])

    old = json.loads((ANALYSIS / 'fixed_workload_resources.json').read_text(encoding='utf-8'))
    res = old['fixed_workload_resources']
    keys = [f'E{i}' for i in range(1, 8)]
    means = {k: report['summaries'][k]['all31']['mean'] for k in keys}
    sds = {k: report['summaries'][k]['all31']['sample_sd'] for k in keys}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, field, label in zip(axes, ['allocated_gb', 'seconds_per_100_batches', 'trainable_parameters'],
                                ['Peak allocated training memory (GB)', 'Training seconds / 100 batches',
                                 'Trainable parameters (millions; log)']):
        xx = {k: res[k][field] for k in keys}
        if field == 'trainable_parameters':
            xx = {k: v/1e6 for k, v in xx.items()}; ax.set_xscale('log')
        frontier = [k for k in keys if not any(xx[j] <= xx[k] and means[j] >= means[k]
                    and (xx[j] < xx[k] or means[j] > means[k]) for j in keys if j != k)]
        frontier.sort(key=xx.get)
        ax.plot([xx[k] for k in frontier], [means[k] for k in frontier], '--', color='gray', lw=.8)
        for k in keys:
            giant = k in ('E4', 'E5', 'E7')
            ax.errorbar(xx[k], means[k], yerr=sds[k], fmt='s' if giant else 'o',
                        color='#c27619' if giant else '#17669b', capsize=2, ms=4)
            ax.annotate(SHORT[k], (xx[k], means[k]), xytext=(5, -10 if k == 'E5' else 5), textcoords='offset points',
                        fontsize=7.5)
        ax.set_xlabel(label, fontsize=8); ax.margins(x=.2); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean test-volume Dice (sample SD)')
    fig.tight_layout(); save(fig, FIGURES[2])


TRAINABLE = {'E0': 655329, 'E1': 655329, 'E2': 305023969, 'E3': 51048417, 'E4': 2260961, 'E5': 786401,
             'E6': 1245153, 'E7': 114132961}
TOTAL = {'E0': 305023969, 'E1': 305023969, 'E2': 305023969, 'E3': 305023969, 'E4': 1138741729,
         'E5': 1137267169, 'E6': 305613793, 'E7': 1137267169}
ORDER = [('Large', 'Frozen', 'E1'), ('Large', 'Partial FT', 'E3'), ('Large', 'LoRA', 'E6'),
         ('Large', 'Full FT', 'E2'), ('Giant', 'Frozen', 'E5'), ('Giant', 'Partial FT', 'E7'), ('Giant', 'LoRA', 'E4')]
GPU_SHORT = {'NVIDIA GeForce RTX 4090': 'RTX 4090', 'NVIDIA GeForce RTX 5090': 'RTX 5090',
             'NVIDIA RTX 4000 Ada Generation': 'RTX 4000 Ada',
             'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb': 'RTX PRO 6000 (MIG 2g.48gb)'}


def run_meta(folder):
    """Validation trajectory, skipped updates, exploratory HD95 and recorded environment of one run."""
    if (folder / 'history.csv').exists():
        with (folder / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        val = [float(r['val_dice_3d_patient_mean']) for r in rows]
        skipped = (sum(int(float(r['skipped_optimizer_updates'])) for r in rows)
                   if rows and 'skipped_optimizer_updates' in rows[0] else None)
    else:
        rows = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        val = [float(r['validation_dice']) for r in rows]
        skipped = sum(int(r['skipped_updates']) for r in rows)
    best = int(np.argmax(val)) + 1  # checkpoints are replaced only on a strict increase: first maximum
    mdd = max(max(val[:t + 1]) - val[t] for t in range(len(val)))
    test = json.loads((folder / 'test_summary.json').read_text(encoding='utf-8'))
    prov = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    weights = prov.get('pretrained_weights', {}).get('sha256') if isinstance(prov.get('pretrained_weights'), dict) \
        else prov.get('pretrained_sha256')
    source = prov.get('dinov2_source_revision') or ('file-sha256:' + prov['dino_source_sha256']
                                                    if prov.get('dino_source_sha256') else 'unknown')
    return {'epochs': len(val), 'best_epoch': best, 'best_val': max(val), 'mdd': mdd, 'btf': max(val) - val[-1],
            'skipped_updates': skipped, 'hd95': test.get('hd95_header_units', {}).get('mean'),
            'gpu': GPU_SHORT.get(prov.get('gpu', ''), prov.get('gpu', 'unknown')), 'torch': prov.get('torch'),
            'python': prov.get('python'), 'source': source, 'weights': weights}


def all_meta():
    folders = {**MAIN, **EXTRA}
    meta = {k: [run_meta(Path(str(p).format(s=s))) for s in SEEDS] for k, p in folders.items()}
    meta['E2_NOCLIP_R'] = [run_meta(SINGLE['E2_NOCLIP_R'])]
    return meta


def gen_tables(report, meta):
    s = report['summaries']
    res = json.loads((ANALYSIS / 'fixed_workload_resources.json').read_text(encoding='utf-8'))['fixed_workload_resources']

    def pm(x, sd, nd=4):
        return f'\\pmv{{{x:.{nd}f}}}{{{sd:.{nd}f}}}'
    out = {}
    # Main accuracy table.
    lines = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\small',
             r'\caption{Native-resolution volume-level test performance under the main protocol (all DINOv2 rows use one '
             r'FP32 loss implementation). Values are mean $\pm$ sample SD across three training seeds. Rows above the '
             r'first rule use $0.5\mathcal{L}_{\mathrm{BCE}}+0.5\mathcal{L}_{\mathrm{Dice}}$; bold marks the highest mean '
             r'within each backbone block, not statistical significance. The controls below use the same split and '
             r'evaluator: the BCE-only frozen objective, and ViT-L trained from random initialization with the Large '
             r'Full FT protocol (no DINOv2 weights). nnU-Net uses Dice+CE and a distinct 3D recipe.}',
             r'\label{tab:accuracy}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcc@{}}', r'\toprule',
             r'Backbone & Adaptation & Dice $\uparrow$ & IoU $\uparrow$\\ \midrule']
    best = {'Large': 'E2', 'Giant': 'E4'}
    for i, (bb, ad, k) in enumerate(ORDER):
        a = s[k]['all31']
        dice, iou = pm(a['mean'], a['sample_sd']), pm(a['iou_mean'], a['iou_sample_sd'])
        if best[bb] == k:
            dice = f"$\\mathbf{{{a['mean']:.4f}}}\\pm{a['sample_sd']:.4f}$"
            iou = f"$\\mathbf{{{a['iou_mean']:.4f}}}\\pm{a['iou_sample_sd']:.4f}$"
        lines.append(f'{bb} & {ad} & {dice} & {iou}\\\\' + (r' \midrule' if i == 3 else ''))
    lines.append(r'\midrule')
    for label, k in [('Large & Frozen, BCE only', 'E0'), ('Large & Full FT, random initialization', 'RAND'),
                     ('Large & Full FT, random init., LR $10^{-4}$ + clip', 'RAND_TUNED')]:
        a = s[k]['all31']
        lines.append(f"{label} & {pm(a['mean'], a['sample_sd'])} & {pm(a['iou_mean'], a['iou_sample_sd'])}\\\\")
    a = s['nnU-Net']['all31']
    lines += [r'\midrule', r'\multicolumn{2}{l}{nnU-Net ResEncM 3D (reference)} & '
              f"{pm(a['mean'], a['sample_sd'])} & {pm(a['iou_mean'], a['iou_sample_sd'])}\\\\",
              r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    out['tab_accuracy.tex'] = '\n'.join(lines) + '\n'
    # Per-seed and sensitivity tables.
    seeds = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\small',
             r'\caption{Mean test-volume Dice for each training seed under the main protocol. Controls and the nnU-Net '
             r'reference are separated.}', r'\label{tab:seeds}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc@{}}', r'\toprule',
             r'Backbone & Adaptation & Seed 42 & Seed 2026 & Seed 3407\\ \midrule']
    sens = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
            r'\begin{table}[!htbp]', r'\centering\small',
            r'\caption{Post-hoc exclusion sensitivity. Values are mean $\pm$ sample SD over the three seed-wise volume '
            r'means.}', r'\label{tab:sensitivity}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcc@{}}',
            r'\toprule', r'Backbone & Adaptation & All 31 volumes & Remaining 30 volumes\\ \midrule']
    rows = [(bb, ad, k) for bb, ad, k in ORDER] + [('Large', 'Frozen, BCE only', 'E0'),
                                                    ('Large', 'Full FT, random init.', 'RAND'),
                                                    ('nnU-Net', 'ResEncM 3D', 'nnU-Net')]
    for i, (bb, ad, k) in enumerate(rows):
        sep = r' \midrule' if i in (3, 6, 8) else ''
        sm = s[k]['all31']['seed_means']
        seeds.append(f'{bb} & {ad} & ' + ' & '.join(f'{v:.4f}' for v in sm) + '\\\\' + sep)
        a, b = s[k]['all31'], s[k]['exclude30']
        sens.append(f"{bb} & {ad} & {pm(a['mean'], a['sample_sd'])} & {pm(b['mean'], b['sample_sd'])}\\\\" + sep)
    seeds += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    sens += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    out['tab_seeds.tex'] = '\n'.join(seeds) + '\n'
    out['tab_sensitivity.tex'] = '\n'.join(sens) + '\n'
    # Efficiency table (fixed-workload memory; GPU-hour estimates from completed epochs).
    eff = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
           r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{Trainable parameters, peak allocated training memory and estimated training cost under the main '
           r'protocol. Peak memory is the largest allocated peak of the fixed-workload benchmark '
           r'(Table~\ref{tab:fixedworkload}). GPU-hours are estimates on the RTX 4090 reference: measured fixed-workload '
           r'seconds per training batch $\times$ 2,376 batches per epoch $\times$ epochs completed, in total or up to the '
           r'validation-selected checkpoint; mean $\pm$ SD across seeds, where the SD reflects only the number of epochs '
           r'completed before early stopping. Validation, checkpoint writing and test inference are excluded.}',
           r'\label{tab:efficiency}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrccc@{}}', r'\toprule',
           r'Backbone & Adaptation & \shortstack{Trainable\\parameters} & \shortstack{Peak memory\\(GB)} & '
           r'\shortstack{GPU-hours\\(total)} & \shortstack{GPU-hours\\(to best)}\\ \midrule']

    def hours(k, field):
        sec = res['E1' if k == 'E0' else k]['seconds_per_100_batches'] / 100
        v = [sec * 2376 * m[field] / 3600 for m in meta[k]]
        return float(np.mean(v)), float(np.std(v, ddof=1))
    for i, (bb, ad, k) in enumerate(ORDER):
        ht, hb = hours(k, 'epochs'), hours(k, 'best_epoch')
        eff.append(f"{bb} & {ad} & {TRAINABLE[k]:,} & {res[k]['allocated_gb']:.3f} & {pm(*ht, nd=2)} & "
                   f"{pm(*hb, nd=2)}\\\\" + (r' \midrule' if i == 3 else ''))
    h0t, h0b = hours('E0', 'epochs'), hours('E0', 'best_epoch')
    eff += [r'\bottomrule', r'\end{tabular*}',
            r'\par\smallskip{\raggedright\footnotesize The BCE-only frozen control has the Large Frozen architecture '
            f"(655,329 trainable parameters, {res['E1']['allocated_gb']:.3f} GB); using the Large Frozen throughput, its "
            f"estimated cost is {pm(*h0t, nd=2)} GPU-hours in total and {pm(*h0b, nd=2)} to the selected checkpoint.\\par}}",
            r'\end{table}']
    out['tab_efficiency.tex'] = '\n'.join(eff) + '\n'
    # Training-length table.
    tl = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
          r'\begin{table}[H]', r'\centering\small',
          r'\caption{Parameter counts and completed training lengths under the main protocol. Epoch triplets follow seed '
          r'order 42 / 2026 / 3407. Total counts include the encoder, decoder and adapters where applicable.}',
          r'\label{tab:traininglength}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrc@{}}', r'\toprule',
          r'Backbone & Adaptation & Total parameters & Trainable (\%) & Epochs\\ \midrule']
    for i, (bb, ad, k) in enumerate(ORDER + [('Large', 'Frozen, BCE only', 'E0'), ('Large', 'Full FT, random init.', 'RAND')]):
        kk = 'E2' if k == 'RAND' else k
        tl.append(f"{bb} & {ad} & {TOTAL[kk]:,} & {100*TRAINABLE[kk]/TOTAL[kk]:.3f} & "
                  + ' / '.join(str(m['epochs']) for m in meta[k]) + '\\\\' + (r' \midrule' if i == 6 else ''))
    tl += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    out['tab_traininglength.tex'] = '\n'.join(tl) + '\n'
    return out


def audit_supplement(report, meta):
    s = report['summaries']
    o = report['other_effects']['E0 - E1 (objective ablation)']['all31']
    names = {k: f'{bb} {ad}' for bb, ad, k in ORDER}
    names.update({'E0': 'Large Frozen (BCE)', 'RAND': 'Large Full FT, random init.',
                  'RAND_TUNED': 'Random init., LR $10^{-4}$ + clip', 'E2_CLIP': 'Large Full FT + clipping',
                  'E2_NOCLIP_R': 'Large Full FT, unclipped rerun', 'V2_FR': 'Common improved: Frozen',
                  'V2_PA': 'Common improved: Partial FT', 'V2_LO': 'Common improved: LoRA',
                  'V2_FU': 'Common improved: Full FT'})
    hd = [r'Backbone / configuration & HD95 (header units)\\ \midrule']
    for k in [k for _, _, k in ORDER] + ['E0']:
        v = [m['hd95'] for m in meta[k]]
        hd.append(f'{names[k]} & \\pmv{{{np.mean(v):.4f}}}{{{np.std(v, ddof=1):.4f}}}\\\\')
    st = []
    for k in [k for _, _, k in ORDER] + ['E0', 'RAND', 'RAND_TUNED', 'E2_CLIP', 'E2_NOCLIP_R', 'V2_FR', 'V2_PA', 'V2_LO', 'V2_FU']:
        for seed, m in zip(SEEDS, meta[k]):
            st.append(f"{names[k]} & {seed} & {m['epochs']} & {m['best_epoch']} & {m['mdd']:.4f} & {m['btf']:.4f}\\\\")
    sk = {k: [m['skipped_updates'] for m in meta[k]] for k in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']}
    missing = [k for k, v in sk.items() if any(x is None for x in v)]
    skip_txt = '; '.join(f"{k}: {'/'.join(str(x) for x in v)}" for k, v in sk.items() if k not in missing)
    # Hardware and software register: one row per configuration (seeds share the listed environment unless noted).
    reg_keys = [('E0', 'Large Frozen (BCE)'), ('E1', 'Large Frozen'), ('E3', 'Large Partial FT'), ('E6', 'Large LoRA'),
                ('E2', 'Large Full FT'), ('E5', 'Giant Frozen'), ('E7', 'Giant Partial FT'), ('E4', 'Giant LoRA'),
                ('RAND', 'Large Full FT, random init.'), ('RAND_TUNED', 'Random init., LR $10^{-4}$ + clip'),
                ('LORA_L_1e-4', 'Large LoRA, LR $10^{-4}$'), ('LORA_G_1e-4', 'Giant LoRA, LR $10^{-4}$'),
                ('E2_CLIP', 'Large Full FT + clipping'), ('E2_NOCLIP_R', 'Large Full FT, unclipped rerun'),
                ('F0', 'Readout: Frozen, last layer'),
                ('F1', 'Readout: Frozen, four layers'), ('L0', 'Readout: LoRA, last layer'),
                ('L1', 'Readout: LoRA, four layers'), ('V2_FR', 'Common improved: Large Frozen'),
                ('V2_PA', 'Common improved: Large Partial FT'), ('V2_LO', 'Common improved: Large LoRA'),
                ('V2_FU', 'Common improved: Large Full FT')]
    sources, weights = {}, {}

    def tag(store, prefix, value):
        if value not in store:
            store[value] = f'{prefix}{len(store) + 1}'
        return store[value]
    reg = []
    for k, label in reg_keys:
        ms = meta[k]
        envs = sorted({(m['gpu'], m['torch'], tag(sources, 'S', m['source']), tag(weights, 'W', m['weights'])) for m in ms})
        for gpu, torch_v, src, w in envs:
            which = [str(sd) for sd, m in zip(SEEDS if len(ms) == 3 else (42,), ms)
                     if (m['gpu'], m['torch'], tag(sources, 'S', m['source']), tag(weights, 'W', m['weights'])) == (gpu, torch_v, src, w)]
            reg.append(f"{label} & {', '.join(which)} & {gpu} & {torch_v} & {src} & {w}\\\\")
    txt = [r'% Generated by scripts/analyze_revision_v2_20261021.py; do not edit by hand.',
           r'\subsection*{Experiment identifiers}', '',
           r'The identifiers link the manuscript to archived runs: E0 = Large Frozen (BCE only); E1 = Large Frozen; '
           r'E2 = Large Full FT; E3 = Large Partial FT; E4 = Giant LoRA; E5 = Giant Frozen; E6 = Large LoRA; E7 = Giant '
           r'Partial FT (blocks 36--39). E1--E7 use the common BCE--Dice objective. E0, E1, E5 and E6 are the reruns with '
           r'the FP32 loss implementation used by all other configurations; the earlier runs of these four '
           r'configurations are not used in any reported result.', '',
           r'\subsection*{Frozen-feature objective sensitivity}', '',
           f"The BCE-only control (E0) has mean Dice {s['E0']['all31']['mean']:.4f}, compared with "
           f"{s['E1']['all31']['mean']:.4f} for the combined objective (E1). The paired difference (E0 minus E1) is "
           f"{o['difference_pp']:.3f} percentage points (95\\% CI [{o['ci95_pp'][0]:.3f}, {o['ci95_pp'][1]:.3f}]), using "
           r'the same seed-averaged case bootstrap as the main analysis. This post-hoc comparison is limited to Large '
           r'frozen features and changes the BCE coefficient as well as the Dice term.', '',
           r'\subsection*{Exploratory surface summaries}', '',
           r'Physical spacing has not been independently verified. The summaries below use image-header units and are '
           r'provided without best-value highlighting; they do not support physical-distance or boundary-superiority '
           r'claims.', '',
           r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{Exploratory HD95 summaries in header units; mean $\pm$ sample SD across seeds.}', r'\label{tab:surface}',
           r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lc@{}}', r'\toprule'] + hd + [
           r'\bottomrule', r'\end{tabular*}', r'\end{table}', '',
           r'\subsection*{Validation-trajectory summaries}', '',
           r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{Post-hoc validation stability on the full recorded trajectory. MDD (maximum drawdown) is the '
           r'largest fall of validation Dice below an earlier value; BTF (best-to-final deterioration) is the best minus '
           r'the final validation Dice. Both are Dice differences on the 0--1 scale. Best epoch is derived from the '
           r'validation history. Observation lengths differ.}',
           r'\label{tab:stability}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
           r'Configuration & Seed & Epochs & Best epoch & MDD & BTF\\ \midrule'] + st + [
           r'\bottomrule', r'\end{tabular*}', r'\end{table}', '',
           f'Skipped optimizer updates over the complete runs (seeds 42/2026/3407): {skip_txt}.'
           + (f" Counters are not available for {', '.join(missing)}." if missing else '')
           + r' These counts are descriptive and are not used to explain the observed declines.', '',
           r'\subsection*{Image-similarity audit}', '',
           r'All 11,781 pairs were screened on a $64\times64\times44$ grid using linear image interpolation and '
           r'nearest-neighbor mask sampling, without registration. Exploratory screening thresholds were Pearson '
           r'correlation $\geq0.95$ or mask Dice $\geq0.90$, with exact binary-mask matches also retained. The union with '
           r'the top 20 pairs by each screening score contained 36 candidates. Of these, 29 shape-matched pairs were '
           r'compared at native resolution; seven shape-mismatched pairs had screening-grid comparisons only. Native '
           r'strong-similarity criteria were correlation $\geq0.99$ over the union of nonzero image voxels and mask Dice '
           r'$\geq0.98$. The identified train--test pair had identical binary masks and image correlation 0.996420. These '
           r'criteria identify image similarity, not patient identity, and were not prespecified patient-matching rules.', '',
           r'\subsection*{Hardware and software register}', '',
           r'Experiments were run on shared institutional GPUs. The register below is transcribed from each run\textquoteright s '
           r'recorded provenance. Inference of the same checkpoint on an RTX 4090 and an RTX 5090 gave Dice identical to '
           r'four decimals; retraining the same configuration and seed on the two GPUs differed by 0.40 percentage points '
           r'(Large LoRA, seed 42: the reported RTX 4090 run and the RTX 5090 run of the single-seed adapter learning-rate '
           r'sweep, which used PyTorch 2.11.0+cu128 and sources S1 and W1). Two further cross-GPU pairs, which also differ in '
           r'source snapshot, differed by 0.38 (Large Frozen) and 0.60 points (Large LoRA; seed 3407, main-table run versus '
           r'last-layer readout cell). These differences are of the order of seed-to-seed variation. Effects of this size or smaller are interpreted with caution. A seed does not '
           r'guarantee bitwise determinism across hardware or software. S1 and S2 are hashes of the DINOv2 Python source '
           r'tree; S3 was recorded by the readout runner as a digest of per-file hashes and is not directly comparable '
           r'with S1 and S2. The nnU-Net reference was trained on an RTX 4090 with PyTorch 2.11.0+cu126 and nnunetv2 '
           r'2.8.1 for all three seeds.', '',
           r'\begin{table}[!htbp]', r'\centering\scriptsize',
           r'\caption{Recorded hardware and software for every reported training run. S: DINOv2 source identifier; '
           r'W: pretrained-weight hash (identifiers listed below the table).}', r'\label{tab:provenance}',
           r'\begin{tabularx}{\linewidth}{@{}Xllllc@{}}', r'\toprule',
           r'Configuration & Seeds & GPU & PyTorch & Source & Weights\\ \midrule'] + reg + [
           r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    for value, t in list(sources.items()) + list(weights.items()):
        txt.append(f'\\noindent {t}: {{\\footnotesize\\nolinkurl{{{value}}}}}\\par')
        txt.append('')
    return '\n'.join(txt) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', required=True)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    out = Path(args.paper_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = analyze()
    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'paired_effects.tex').write_text(paired_table(report), encoding='utf-8')
    (out / 'recipe_external_tables.tex').write_text(recipe_external_tables(report), encoding='utf-8')
    (out / 'tab_protocol_v2.tex').write_text(protocol_v2_table(report), encoding='utf-8')
    contrasts_fig, pareto_fig = figure_envs()
    (out / 'revision_analysis_20261006.tex').write_text(contrasts_fig, encoding='utf-8')
    (out / 'fig_pareto_revision.tex').write_text(pareto_fig, encoding='utf-8')
    meta = all_meta()
    for name, content in gen_tables(report, meta).items():
        (out / name).write_text(content, encoding='utf-8')
    (out / 'audit_supplement.tex').write_text(audit_supplement(report, meta), encoding='utf-8')
    report['run_meta'] = {k: v for k, v in meta.items()}
    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    if args.figures:
        plot(report, out)
    s = report['summaries']
    print(json.dumps({k: [round(s[k]['all31']['mean'], 4), s[k]['all31']['sample_sd'] and round(s[k]['all31']['sample_sd'], 4),
                          round(s[k]['all31']['iou_mean'], 4), round(s[k]['exclude30']['mean'], 4)] for k in s}, indent=0))
    for n, e in report['rq1_paired_effects'].items():
        a = e['all31']
        print(f"{n:36s} {a['difference_pp']:+.3f} [{a['ci95_pp'][0]:+.3f},{a['ci95_pp'][1]:+.3f}] p={a['p_two_sided_bootstrap']:.4f} holm={a['p_holm_16']:.4f} seeds={[round(v,2) for v in e['seed_mean_differences_pp']]}")
    for n, e in report['other_effects'].items():
        a = e.get('all31', e)
        print(f"{n:36s} {a['difference_pp']:+.3f} median={a['median_pp']:+.2f} pos={a['positive_cases']}")


if __name__ == '__main__':
    main()
