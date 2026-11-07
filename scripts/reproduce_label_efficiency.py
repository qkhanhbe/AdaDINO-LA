"""Label-budget analysis (3/11/2026): the common improved recipe and nnU-Net ResEncM 3D trained on the
pre-specified nested subsets of 11 and 27 of the 108 training volumes, against the full-data runs.

Inputs are read-only: results/label_efficiency_20261027 (reduced-data runs, the random-initialization cell at
100% and the LoRA seed-42 parity run), results/revision_v2_20261021/protocol_v2 (full-data common recipe) and
the nnU-Net reference in output/analysis. The Holm family (contrasts at 11 and 27 volumes and their change from
108 to 11 volumes) was defined after the reduced-data test results had been viewed, so all tests are exploratory.

Run: python -B scripts/reproduce_label_efficiency.py --paper-dir <overleaf working copy>
Writes outputs/label_efficiency.json and, in --paper-dir, tab_labels.tex.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_revision_v2 import bootstrap, digest, fmt_p, holm, num  # noqa: E402

LE = ROOT / 'results/label_efficiency'
V2 = ROOT / 'results/revision_v2/protocol_v2'
NN100 = ROOT / 'results/nnUNet_ResEncM_3D'
JSON_PATH = ROOT / 'outputs/label_efficiency.json'
SEEDS = (42, 2026, 3407)
FRACTIONS = {10: 11, 25: 27, 100: 108}  # percent -> training volumes
METHODS = {'V_SC': 'Random init.\\ (Full FT)', 'V_FR': 'Frozen', 'V_PA': 'Partial FT', 'V_LO': 'LoRA',
           'V_FU': 'Full FT', 'nnU-Net': '\\mbox{nnU-Net} ResEncM 3D'}
CONTRASTS = {  # name -> weights over methods, evaluated within one fraction
    'Full FT - random init.': {'V_FU': 1, 'V_SC': -1},
    'LoRA - Full FT': {'V_LO': 1, 'V_FU': -1},
    'LoRA - Frozen': {'V_LO': 1, 'V_FR': -1},
    'nnU-Net - LoRA': {'nnU-Net': 1, 'V_LO': -1},
    'nnU-Net - Full FT': {'nnU-Net': 1, 'V_FU': -1},
}
LABELS = {'Full FT - random init.': 'Full FT $-$ random init.', 'LoRA - Full FT': 'LoRA $-$ Full FT',
          'LoRA - Frozen': 'LoRA $-$ Frozen', 'nnU-Net - LoRA': '\\mbox{nnU-Net} $-$ LoRA',
          'nnU-Net - Full FT': '\\mbox{nnU-Net} $-$ Full FT'}
CHANGE = (10, 100)  # change column: value with 11 volumes minus value with 108 volumes


def analyze():
    sources = {}

    def track(path):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return path

    ids = sorted(track(ROOT / 'splits/test_patients.txt').read_text().split())
    assert len(ids) == len(set(ids)) == 31

    def from_csv(path, seed):
        with track(path).open(encoding='utf-8-sig', newline='') as stream:
            rows = {r['patient_id']: r for r in csv.DictReader(stream)}
        assert sorted(rows) == ids and all(int(r['seed']) == seed for r in rows.values()), path
        return [float(rows[i]['dice_3d']) for i in ids], [float(rows[i]['iou_3d']) for i in ids]

    def from_json(path, seed):
        rows = {r['case_id']: r for r in json.loads(track(path).read_text(encoding='utf-8'))}
        assert sorted(rows) == ids and all(int(r['seed']) == seed for r in rows.values()), path
        return [float(rows[i]['dice_3d']) for i in ids], [float(rows[i].get('iou_3d', np.nan)) for i in ids]

    def load(method, fraction, seed):
        if method == 'nnU-Net':
            if fraction == 100:
                return from_json(NN100 / f'seed{seed}/test_case_metrics.json', seed)
            return from_json(LE / f'nnunet/le_f{fraction:03d}_seed{seed}/test_case_metrics.json', seed)
        if fraction == 100 and method != 'V_SC':
            return from_csv(V2 / f'{method}/seed{seed}/test_patient_metrics.csv', seed)
        return from_csv(LE / f'dino/f{fraction:03d}/{method}/seed{seed}/test_patient_metrics.csv', seed)

    dice, iou = {}, {}
    for m in METHODS:
        for f in FRACTIONS:
            pairs = [load(m, f, s) for s in SEEDS]
            dice[m, f] = np.array([p[0] for p in pairs])
            iou[m, f] = np.array([p[1] for p in pairs])
            assert np.isfinite(dice[m, f]).all() and ((0 <= dice[m, f]) & (dice[m, f] <= 1)).all()

    training = {}
    for m in METHODS:
        for f in FRACTIONS:
            if m == 'nnU-Net' or (f == 100 and m != 'V_SC'):
                continue
            for s in SEEDS:
                t = json.loads(track(LE / f'dino/f{f:03d}/{m}/seed{s}/train_complete.json').read_text())
                training[f'{m} f{f:03d} seed{s}'] = {k: t[k] for k in ('epochs', 'best_epoch', 'best_validation_dice')}

    summaries = {}
    for (m, f), d in dice.items():
        sm = d.mean(axis=1)
        im = iou[m, f].mean(axis=1)
        summaries[f'{m} f{f:03d}'] = {'mean': float(sm.mean()), 'sample_sd': float(sm.std(ddof=1)),
                                      'seed_means': sm.tolist(), 'iou_mean': float(np.nanmean(im))}

    def contrast(weights):  # weights over (method, fraction)
        per_seed_case = sum(w*dice[k] for k, w in weights.items())
        # Case x seed bootstrap: all configurations of a seed share its subset, so the three seeds (and with
        # them the subsets) are resampled jointly, then the cases; the interval includes subset variation.
        rng = np.random.default_rng(20260926)
        seeds = rng.integers(0, len(SEEDS), (20000, len(SEEDS)))
        cases = rng.integers(0, len(ids), (20000, len(ids)))
        per_case = per_seed_case[seeds].mean(axis=1)
        joint = np.take_along_axis(per_case, cases, axis=1).mean(axis=1)
        return {**bootstrap(per_seed_case.mean(axis=0)),
                'seed_mean_differences_pp': (100*per_seed_case.mean(axis=1)).tolist(),
                'case_seed_ci95_pp': (100*np.quantile(joint, [.025, .975])).tolist()}

    effects = {}
    for name, w in CONTRASTS.items():
        for f in FRACTIONS:
            effects[f'{name} @{FRACTIONS[f]}'] = contrast({(m, f): x for m, x in w.items()})
        a, b = CHANGE
        effects[f'{name} change {FRACTIONS[a]}-{FRACTIONS[b]}'] = contrast(
            {**{(m, a): x for m, x in w.items()}, **{(m, b): -x for m, x in w.items()}})
    family = [n for n in effects if not n.endswith(f'@{FRACTIONS[100]}')]
    adjusted = holm({n: effects[n]['p_two_sided_bootstrap'] for n in family})
    for n in family:
        effects[n]['p_holm'] = adjusted[n]
    drops = {m: contrast({(m, CHANGE[0]): 1, (m, CHANGE[1]): -1}) for m in METHODS}
    # Across budgets (unadjusted): does a pipeline with 11 volumes reach another trained on all 108?
    across = {'Full FT @11 - random init. @108': contrast({('V_FU', 10): 1, ('V_SC', 100): -1}),
              'nnU-Net @11 - LoRA @108': contrast({('nnU-Net', 10): 1, ('V_LO', 100): -1}),
              'nnU-Net @11 - Full FT @108': contrast({('nnU-Net', 10): 1, ('V_FU', 100): -1})}
    parity = {'archived_v2_V_LO_seed42': float(np.mean(load('V_LO', 100, 42)[0])),
              'rerun_V_LO_seed42': float(np.mean(from_csv(LE / 'dino/f100/V_LO/seed42/test_patient_metrics.csv', 42)[0]))}
    return {'bootstrap': {'resamples': 20000, 'seed': 20260926, 'reset_rng_each_contrast': True,
                          'holm_family': family, 'holm_family_status': 'defined after the reduced-data results were viewed'},
            'fractions_percent_to_volumes': FRACTIONS, 'case_ids': ids, 'summaries': summaries, 'effects': effects,
            'drop_11_minus_108': drops, 'across_budgets': across, 'training': training, 'parity': parity,
            'source_files': [{'path': k, 'sha256': v} for k, v in sorted(sources.items())],
            'generator_sha256': digest(Path(__file__)), 'numpy_version': np.__version__}


def cell(x):
    return f"${num(x['difference_pp'])}\\;[{num(x['ci95_pp'][0])},\\,{num(x['ci95_pp'][1])}]$"


def labels_table(report):
    s, e, d = report['summaries'], report['effects'], report['drop_11_minus_108']
    v = FRACTIONS
    lines = [r'% Generated by scripts/analyze_label_efficiency_20261103.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Reduced label budgets (RQ1; exploratory, design fixed before any reduced-data run but after the '
             r'main results). ViT-Large configurations use the common improved recipe without augmentation and '
             r'\mbox{nnU-Net} its own recipe with augmentation; each seed trains on one nested random subset of the 108 '
             r'training volumes, shared by all configurations, and every run selects its checkpoint on the same 15 '
             r'annotated validation volumes and is tested on the same 31 volumes. Random init.: the same ViT-Large '
             r'trained from random initialization with the common recipe and an encoder learning rate of $10^{-4}$; its '
             r'108-volume runs are new and therefore differ from both from-scratch rows of Table~\ref{tab:accuracy}. The '
             r'other 108-volume DINOv2 rows are the runs of Table~\ref{tab:v2}. Upper block: Dice, mean $\pm$ SD of three '
             r'seed means. Lower block: paired case-bootstrap differences in percentage points with 95\% CIs; the last '
             r'column is the change of each quantity from 108 to 11 training volumes.}', r'\label{tab:labels}',
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcccc@{}}',
             f'\\toprule Training volumes & {v[10]} (10\\%) & {v[25]} (25\\%) & {v[100]} (100\\%) & '
             f'Change, {v[10]} $-$ {v[100]}\\\\ \\midrule',
             r'\multicolumn{5}{@{}l}{\textit{Dice}}\\']
    for m, label in METHODS.items():
        pm = ' & '.join(f"\\pmv{{{s[f'{m} f{f:03d}']['mean']:.4f}}}{{{s[f'{m} f{f:03d}']['sample_sd']:.4f}}}"
                        for f in FRACTIONS)
        lines.append(f'\\quad {label} & {pm} & {cell(d[m])}\\\\')
    lines += [r'\addlinespace[2pt]', r'\multicolumn{5}{@{}l}{\textit{Paired differences, percentage points}}\\']
    for name, label in LABELS.items():
        cells = [cell(e[f'{name} @{v[f]}']) for f in FRACTIONS] + [cell(e[f'{name} change {v[10]}-{v[100]}'])]
        lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r'\\')
    family = report['bootstrap']['holm_family']
    weak = {}  # contrast label -> [(column, adjusted p)] for adjusted p >= 0.002
    for n in family:
        if e[n]['p_holm'] >= 0.002:
            name, where = (n.split(' @') if ' @' in n else (n.split(' change')[0], 'change'))
            weak.setdefault(LABELS[name], []).append((where if where == 'change' else f'{where} volumes',
                                                      fmt_p(e[n]['p_holm'])))
    note = (f'$p_{{\\mathrm{{Holm}}}}$ over the {len(family)} contrasts at 11 and 27 volumes and in the change column '
            '(family defined after these results were viewed): ' +
            '; '.join(f'{label}, ' + ', '.join(f'{w} {p}' for w, p in cols) for label, cols in weak.items()) +
            '; all others $<0.002$. ')
    # Robustness screen of Section 4.5, extended by a case x seed bootstrap that also resamples the subsets.
    clear = [n for n in e if e[n]['ci95_pp'][0] > 0 or e[n]['ci95_pp'][1] < 0]
    same_sign = all(len({np.sign(x) for x in e[n]['seed_mean_differences_pp']}) == 1 for n in clear)
    joint = all(e[n]['case_seed_ci95_pp'][0] > 0 or e[n]['case_seed_ci95_pp'][1] < 0 for n in clear)
    others = [n for n in e if n not in clear]
    joint_others = all(e[n]['case_seed_ci95_pp'][0] <= 0 <= e[n]['case_seed_ci95_pp'][1] for n in others)
    assert same_sign and joint and joint_others, 'robustness statement below would be false'
    screen = ('Every contrast whose interval excludes zero has the same sign in all three seeds, and a case $\\times$ '
              'seed bootstrap that also resamples the subsets excludes zero for the same contrasts and only for them. ')
    lines += [r'\bottomrule\end{tabularx}', r'\par\smallskip\raggedright\scriptsize',
              note + 'The 108-volume contrasts carry unadjusted intervals. ' + screen + 'Every reduced-data epoch '
              'draws 2,376 batches, as a full-data epoch does, so all budgets share the update schedule and '
              'early-stopping rule. \\mbox{nnU-Net} plans were derived from the 123 development volumes for every budget.',
              r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path)
    args = parser.parse_args()
    report = analyze()
    JSON_PATH.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {JSON_PATH}')
    if args.paper_dir:
        (args.paper_dir / 'tab_labels.tex').write_text(labels_table(report), encoding='utf-8')
        print(f"wrote {args.paper_dir / 'tab_labels.tex'}")


if __name__ == '__main__':
    main()
