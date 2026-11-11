"""Augmentation study (pre-specified 10/11/2026, post hoc relative to the test set): the nine planned contrasts.

LoRA (DINOv2 ViT-L/14, common recipe) and the randomly initialized ViT-L/14 (full fine-tuning, encoder LR 1e-4), both
trained with nnU-Net-style 2D augmentation on 108 and 11 training volumes (seeds 42/2026/3407, the nested subsets of
the label-budget study). References without augmentation: the label-budget runs (random init. at 108 and 11 volumes,
LoRA at 11) and the common-recipe LoRA runs at 108, as in the label-budget analysis; nnU-Net 3D at 108 and 11.

Contrasts (Holm over all nine): pretraining gain with augmentation (LoRA_aug - scratch_aug) at 108, at 11 and its
change; augmentation effect without pretraining (scratch_aug - scratch) at 108 and 11; augmentation effect with LoRA
(LoRA_aug - LoRA) at 108 and 11; remaining nnU-Net gap (nnU-Net - LoRA_aug) at 108 and 11. Same paired case bootstrap
as the main analysis (seeds averaged within case, 20,000 resamples, RNG seed 20260926 reset per contrast) plus the
case x seed bootstrap of the label-budget analysis (seeds, and with them the subsets, resampled jointly).

Run: python -B scripts/reproduce_augmentation_study.py [--paper-dir <manuscript working copy>]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_revision_v2 import bootstrap, digest, fmt_p, holm, num  # noqa: E402

AUG = ROOT / 'results/augmentation_study'
LE = ROOT / 'results/label_efficiency'
V2 = ROOT / 'results/revision_v2/protocol_v2'
NN100 = ROOT / 'results/nnUNet_ResEncM_3D'
JSON_PATH = ROOT / 'outputs/augmentation_study.json'
SEEDS = (42, 2026, 3407)
VOLUMES = {100: 108, 10: 11}
ARMS = {'LoRA_aug': 'LoRA, augmented', 'LoRA': 'LoRA', 'SC_aug': 'Random init., augmented', 'SC': 'Random init.',
        'nnU-Net': '\\mbox{nnU-Net} ResEncM 3D'}
CONTRASTS = [  # name, weights, budget(s)
    ('Pretraining gain, augmented (LoRA - random init.)', {'LoRA_aug': 1, 'SC_aug': -1}),
    ('Augmentation effect, random init.', {'SC_aug': 1, 'SC': -1}),
    ('Augmentation effect, LoRA', {'LoRA_aug': 1, 'LoRA': -1}),
    ('nnU-Net - augmented LoRA', {'nnU-Net': 1, 'LoRA_aug': -1}),
]


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
        return [float(rows[i]['dice_3d']) for i in ids]

    def from_json(path, seed):
        rows = {r['case_id']: r for r in json.loads(track(path).read_text(encoding='utf-8'))}
        assert sorted(rows) == ids and all(int(r['seed']) == seed for r in rows.values()), path
        return [float(rows[i]['dice_3d']) for i in ids]

    def load(arm, frac, seed):
        if arm == 'LoRA_aug':
            return from_csv(AUG / f'dino/f{frac:03d}/V_LO/seed{seed}/test_patient_metrics.csv', seed)
        if arm == 'SC_aug':
            return from_csv(AUG / f'dino/f{frac:03d}/V_SC/seed{seed}/test_patient_metrics.csv', seed)
        if arm == 'SC':
            return from_csv(LE / f'dino/f{frac:03d}/V_SC/seed{seed}/test_patient_metrics.csv', seed)
        if arm == 'LoRA':
            if frac == 100:
                return from_csv(V2 / f'V_LO/seed{seed}/test_patient_metrics.csv', seed)
            return from_csv(LE / f'dino/f{frac:03d}/V_LO/seed{seed}/test_patient_metrics.csv', seed)
        if frac == 100:
            return from_json(NN100 / f'seed{seed}/test_case_metrics.json', seed)
        return from_json(LE / f'nnunet/le_f{frac:03d}_seed{seed}/test_case_metrics.json', seed)

    dice = {(a, f): np.array([load(a, f, s) for s in SEEDS]) for a in ARMS for f in VOLUMES}
    for d in dice.values():
        assert d.shape == (3, 31) and np.isfinite(d).all() and ((0 <= d) & (d <= 1)).all()

    # identity checks: the augmented runs used the pre-specified protocol, subsets and cells
    training = {}
    for cell, arm in (('V_LO', 'LoRA_aug'), ('V_SC', 'SC_aug')):
        for f in VOLUMES:
            for s in SEEDS:
                run = AUG / f'dino/f{f:03d}/{cell}/seed{s}'
                cfg = json.loads(track(run / 'config.json').read_text(encoding='utf-8'))
                assert cfg['augmentation'] is True and cfg['seed'] == s and cfg['fraction_percent'] == f
                assert cfg['protocol']['name'] == 'augmentation_study_v2' and cfg['train_volumes'] == VOLUMES[f]
                done = json.loads(track(run / 'train_complete.json').read_text(encoding='utf-8'))
                hist = json.loads(track(run / 'history.json').read_text(encoding='utf-8'))
                training[f'{arm} f{f:03d} seed{s}'] = {
                    'epochs': done['epochs'], 'best_epoch': done['best_epoch'],
                    'best_validation_dice': done['best_validation_dice'],
                    'reached_epoch_limit': done['epochs'] == cfg['protocol']['max_epochs'],
                    'skipped_updates': sum(int(h['skipped_updates']) for h in hist)}
                le_cfg = LE / f'dino/f{f:03d}/{cell}/seed{s}/config.json'
                if le_cfg.exists():  # same nested subset as the label-budget run of this seed
                    assert json.loads(track(le_cfg).read_text(encoding='utf-8'))['train_cases'] == cfg['train_cases']

    summaries = {}
    for (a, f), d in dice.items():
        sm = d.mean(axis=1)
        summaries[f'{a} @{VOLUMES[f]}'] = {'mean': float(sm.mean()), 'sample_sd': float(sm.std(ddof=1)),
                                           'seed_means': sm.tolist()}

    def contrast(weights):
        per_seed_case = sum(w * dice[k] for k, w in weights.items())
        rng = np.random.default_rng(20260926)
        seeds = rng.integers(0, len(SEEDS), (20000, len(SEEDS)))
        cases = rng.integers(0, len(ids), (20000, len(ids)))
        per_case = per_seed_case[seeds].mean(axis=1)
        joint = np.take_along_axis(per_case, cases, axis=1).mean(axis=1)
        return {**bootstrap(per_seed_case.mean(axis=0)),
                'seed_mean_differences_pp': (100 * per_seed_case.mean(axis=1)).tolist(),
                'case_seed_ci95_pp': (100 * np.quantile(joint, [.025, .975])).tolist()}

    effects = {}
    for name, w in CONTRASTS:
        for f in (100, 10):
            effects[f'{name} @{VOLUMES[f]}'] = contrast({(a, f): x for a, x in w.items()})
    name, w = CONTRASTS[0]
    effects[f'{name} change 11-108'] = contrast({**{(a, 10): x for a, x in w.items()},
                                                 **{(a, 100): -x for a, x in w.items()}})
    assert len(effects) == 9
    adjusted = holm({n: e['p_two_sided_bootstrap'] for n, e in effects.items()})
    for n in effects:
        effects[n]['p_holm'] = adjusted[n]
    # descriptive (not part of the planned family): the unaugmented pretraining gain for comparison
    reference = {f'Pretraining gain, unaugmented (LoRA - random init.) @{VOLUMES[f]}':
                 contrast({('LoRA', f): 1, ('SC', f): -1}) for f in (100, 10)}
    return {'status': 'pre-specified 2026-11-10 before any augmented run; post hoc relative to the test set',
            'bootstrap': {'resamples': 20000, 'seed': 20260926, 'reset_rng_each_contrast': True,
                          'holm_family': list(effects)},
            'case_ids': ids, 'summaries': summaries, 'effects': effects, 'reference_unadjusted': reference,
            'training': training,
            'source_files': [{'path': k, 'sha256': v} for k, v in sorted(sources.items())],
            'generator_sha256': digest(Path(__file__)), 'numpy_version': np.__version__}


def cell(x):
    return f"${num(x['difference_pp'])}\\;[{num(x['ci95_pp'][0])},\\,{num(x['ci95_pp'][1])}]$"


def table(report):
    s, e, t = report['summaries'], report['effects'], report['training']
    lines = [r'% Generated by scripts/analyze_augmentation_study_20261111.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Augmentation study (post hoc; design and nine contrasts fixed before any augmented run). The '
             r'common improved recipe of the label-budget analysis, with nnU-Net-style 2D augmentation of the training '
             r'slices (rotation, scaling, noise, blur, brightness, contrast, low resolution, gamma, mirroring) as the '
             r'only change; same nested subsets, 50-epoch limit, early stopping, validation and test volumes. Upper '
             r'block: Dice, mean $\pm$ SD of three seed means. Lower block: paired case-bootstrap differences in '
             r'percentage points with 95\% CIs and $p_{\mathrm{Holm}}$ over the nine contrasts.}',
             r'\label{tab:augmentation}',
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xccc@{}}',
             r'\toprule & 11 training volumes & 108 training volumes & Change, 11 $-$ 108\\ \midrule',
             r'\multicolumn{4}{@{}l}{\textit{Dice}}\\']
    for a, label in ARMS.items():
        pm = ' & '.join(f"\\pmv{{{s[f'{a} @{v}']['mean']:.4f}}}{{{s[f'{a} @{v}']['sample_sd']:.4f}}}" for v in (11, 108))
        lines.append(f'\\quad {label} & {pm} & \\\\')
    lines += [r'\addlinespace[2pt]', r'\multicolumn{4}{@{}l}{\textit{Paired differences, percentage points ($p_{\mathrm{Holm}}$)}}\\']
    for name, _ in CONTRASTS:
        cells = []
        for v in (11, 108):
            x = e[f'{name} @{v}']
            cells.append(f"{cell(x)} ({fmt_p(x['p_holm'])})")
        ch = e.get(f'{name} change 11-108')
        cells.append(f"{cell(ch)} ({fmt_p(ch['p_holm'])})" if ch else '')
        label = name.replace(' - ', ' $-$ ').replace('nnU-Net', '\\mbox{nnU-Net}')
        lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r'\\')
    sc_cap = [k for k, v in t.items() if k.startswith('SC_aug') and v['reached_epoch_limit']]
    epochs = '; '.join(f"{lab} {', '.join(str(t[f'{a} f{f:03d} seed{x}']['epochs']) + '/' + str(t[f'{a} f{f:03d} seed{x}']['best_epoch']) for x in SEEDS)}"
                       for a, lab in (('LoRA_aug', 'LoRA'), ('SC_aug', 'random init.')) for f, lab in
                       ((100, f'{lab}, 108 volumes'), (10, f'{lab}, 11 volumes')))
    lines += [r'\bottomrule\end{tabularx}', r'\par\smallskip\raggedright\scriptsize',
              'Epochs completed/selected epoch (seeds 42/2026/3407): ' + epochs + '. '
              f'{len(sc_cap)} of the six augmented random-init.\\ runs reached the 50-epoch limit, so they had not '
              'converged. Unaugmented references: label-budget runs, and the common-recipe LoRA runs at 108 volumes.',
              r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path)
    args = parser.parse_args()
    report = analyze()
    JSON_PATH.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {JSON_PATH}')
    for n, x in report['effects'].items():
        print(f"{n:62s} {x['difference_pp']:+7.2f} [{x['ci95_pp'][0]:+6.2f},{x['ci95_pp'][1]:+6.2f}] "
              f"pHolm={x['p_holm']:.4f} seeds={[round(v, 2) for v in x['seed_mean_differences_pp']]} "
              f"cxs=[{x['case_seed_ci95_pp'][0]:+.2f},{x['case_seed_ci95_pp'][1]:+.2f}] pos={x['positive_cases']}")
    for n, x in report['summaries'].items():
        print(f"{n:22s} {x['mean']:.4f} +/- {x['sample_sd']:.4f}")
    for n, x in report['reference_unadjusted'].items():
        print(f"{n:62s} {x['difference_pp']:+7.2f} [{x['ci95_pp'][0]:+6.2f},{x['ci95_pp'][1]:+6.2f}]")
    if args.paper_dir:
        (args.paper_dir / 'tab_augmentation.tex').write_text(table(report), encoding='utf-8')
        print(f"wrote {args.paper_dir / 'tab_augmentation.tex'}")


if __name__ == '__main__':
    main()
