"""Decoder study analysis (5/11/2026): the common improved recipe with the UNETR-style hierarchical decoder against
the same recipe with the lightweight decoder, and nnU-Net ResEncM 2D as a slice-wise counterpart of the 3D reference.

The planned contrasts were fixed in src/decoder_study.py on 1/11/2026, before any decoder-study run (post hoc with
respect to the main and v2 results): hierarchical minus lightweight decoder per strategy and training budget, the
gap of nnU-Net 3d_fullres to each hierarchical cell, and whether the strategy ranking with 108 volumes changes. Holm
adjustment runs over these twelve contrasts. The nnU-Net 2D contrasts were not planned and carry unadjusted intervals.

Inputs are read-only: results/decoder_study_20261101 (hierarchical-decoder runs and nnU-Net 2D),
results/revision_v2_20261021/protocol_v2 (lightweight decoder, 108 volumes), results/label_efficiency_20261027
(lightweight decoder and nnU-Net 3D, 11 volumes) and the 108-volume nnU-Net 3D reference in output/analysis.

Run: python -B scripts/reproduce_decoder_study.py [--paper-dir <overleaf working copy>]
Writes outputs/decoder_study.json and, in --paper-dir, tab_decoder.tex.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_revision_v2 import bootstrap, digest, fmt_p, holm, num  # noqa: E402

DEC = ROOT / 'results/decoder_study'
LE = ROOT / 'results/label_efficiency'
V2 = ROOT / 'results/revision_v2/protocol_v2'
NN100 = ROOT / 'results/nnUNet_ResEncM_3D'
JSON_PATH = ROOT / 'outputs/decoder_study.json'
SEEDS = (42, 2026, 3407)
FRACTIONS = {10: 11, 100: 108}  # percent -> training volumes
STRATEGIES = {'FR': 'Frozen', 'PA': 'Partial FT', 'LO': 'LoRA', 'FU': 'Full FT'}
CELLS = {10: ('LO', 'FU'), 100: ('FR', 'PA', 'LO', 'FU')}  # hierarchical cells run at each budget
DECODER_PARAMS = {'light': 655_329, 'hierarchical': 3_121_057}
STRATEGY_CONTRASTS = (('FU', 'LO'), ('LO', 'PA'), ('PA', 'FR'), ('FU', 'FR'))


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

    def load(model, fraction, seed):
        if model == 'nnU-Net 3D':
            if fraction == 100:
                return from_json(NN100 / f'seed{seed}/test_case_metrics.json', seed)
            return from_json(LE / f'nnunet/le_f{fraction:03d}_seed{seed}/test_case_metrics.json', seed)
        if model == 'nnU-Net 2D':
            assert fraction == 100
            return from_json(DEC / f'nnunet2d/ref2d_seed{seed}/test_case_metrics.json', seed)
        decoder, strategy = model.split()
        if decoder == 'H':
            return from_csv(DEC / f'dino/f{fraction:03d}/D_{strategy}/seed{seed}/test_patient_metrics.csv', seed)
        if fraction == 100:
            return from_csv(V2 / f'V_{strategy}/seed{seed}/test_patient_metrics.csv', seed)
        return from_csv(LE / f'dino/f{fraction:03d}/V_{strategy}/seed{seed}/test_patient_metrics.csv', seed)

    models = {f: [f'{d} {s}' for s in CELLS[f] for d in ('L', 'H')] + ['nnU-Net 3D'] for f in FRACTIONS}
    models[100].append('nnU-Net 2D')
    dice = {}
    for f, names in models.items():
        for m in names:
            dice[m, f] = np.array([load(m, f, s) for s in SEEDS])
            assert np.isfinite(dice[m, f]).all() and ((0 <= dice[m, f]) & (dice[m, f] <= 1)).all()

    training = {}
    for f in FRACTIONS:
        for s in CELLS[f]:
            for seed in SEEDS:
                run = DEC / f'dino/f{f:03d}/D_{s}/seed{seed}'
                t = json.loads(track(run / 'train_complete.json').read_text())
                assert t['identity'] == {**t['identity'], 'cell': f'D_{s}', 'seed': seed, 'fraction': f}
                training[f'H {s} f{f:03d} seed{seed}'] = {k: t[k] for k in ('epochs', 'best_epoch', 'best_validation_dice')}

    for seed in SEEDS:  # nnU-Net 2D keeps checkpoint_best: the last epoch with a new best EMA pseudo Dice
        epoch = best = None
        for line in track(DEC / f'nnunet2d/ref2d_seed{seed}/training_log.txt').read_text(encoding='utf-8').splitlines():
            if (m := re.search(r': Epoch (\d+)\s*$', line)):
                epoch = int(m.group(1))
            elif 'New best EMA pseudo Dice' in line:
                best = epoch
        # nnU-Net counts epochs from 0; stored 1-based like the DINOv2 runs.
        training[f'nnU-Net 2D f100 seed{seed}'] = {'epochs': epoch + 1, 'best_epoch': best + 1}

    summaries = {}
    for (m, f), d in dice.items():
        sm = d.mean(axis=1)
        summaries[f'{m} f{f:03d}'] = {'mean': float(sm.mean()), 'sample_sd': float(sm.std(ddof=1)),
                                      'seed_means': sm.tolist()}

    def contrast(weights):  # weights over (model, fraction)
        per_seed_case = sum(w*dice[k] for k, w in weights.items())
        # Case x seed bootstrap as in the label-budget analysis: seeds (and with them the subsets) resampled jointly.
        rng = np.random.default_rng(20260926)
        seeds = rng.integers(0, len(SEEDS), (20000, len(SEEDS)))
        cases = rng.integers(0, len(ids), (20000, len(ids)))
        per_case = per_seed_case[seeds].mean(axis=1)
        joint = np.take_along_axis(per_case, cases, axis=1).mean(axis=1)
        return {**bootstrap(per_seed_case.mean(axis=0)),
                'seed_mean_differences_pp': (100*per_seed_case.mean(axis=1)).tolist(),
                'case_seed_ci95_pp': (100*np.quantile(joint, [.025, .975])).tolist()}

    planned = {}
    for f in FRACTIONS:
        for s in CELLS[f]:
            planned[f'H - L {s} @{FRACTIONS[f]}'] = contrast({(f'H {s}', f): 1, (f'L {s}', f): -1})
            planned[f'nnU-Net 3D - H {s} @{FRACTIONS[f]}'] = contrast({('nnU-Net 3D', f): 1, (f'H {s}', f): -1})
    adjusted = holm({n: e['p_two_sided_bootstrap'] for n, e in planned.items()})
    for n in planned:
        planned[n]['p_holm'] = adjusted[n]

    exploratory = {'nnU-Net 3D - nnU-Net 2D @108': contrast({('nnU-Net 3D', 100): 1, ('nnU-Net 2D', 100): -1})}
    for s in CELLS[100]:
        exploratory[f'nnU-Net 2D - H {s} @108'] = contrast({('nnU-Net 2D', 100): 1, (f'H {s}', 100): -1})
    for f in FRACTIONS:
        for s in CELLS[f]:
            exploratory[f'nnU-Net 3D - L {s} @{FRACTIONS[f]}'] = contrast({('nnU-Net 3D', f): 1, (f'L {s}', f): -1})
    # The contrasts of the common-recipe table, under each decoder (the lightweight ones reproduce that table).
    for a, b in STRATEGY_CONTRASTS:
        for d in ('L', 'H'):
            exploratory[f'{d} {a} - {d} {b} @108'] = contrast({(f'{d} {a}', 100): 1, (f'{d} {b}', 100): -1})
    exploratory['H FU - H LO @11'] = contrast({('H FU', 10): 1, ('H LO', 10): -1})
    # Decoder effect with 11 minus with 108 volumes (does the hierarchical decoder help more with fewer labels?)
    for s in CELLS[10]:
        exploratory[f'H - L {s} change 11-108'] = contrast({(f'H {s}', 10): 1, (f'L {s}', 10): -1,
                                                            (f'H {s}', 100): -1, (f'L {s}', 100): 1})

    ranking = {d: sorted(CELLS[100], key=lambda s: -summaries[f'{d} {s} f100']['mean']) for d in ('L', 'H')}
    return {'bootstrap': {'resamples': 20000, 'seed': 20260926, 'reset_rng_each_contrast': True,
                          'holm_family': list(planned), 'holm_family_status': 'planned 1/11/2026 before any run'},
            'fractions_percent_to_volumes': FRACTIONS, 'decoder_parameters': DECODER_PARAMS, 'case_ids': ids,
            'summaries': summaries, 'planned': planned, 'exploratory': exploratory,
            'ranking_at_108_by_mean': ranking, 'ranking_changed': ranking['L'] != ranking['H'],
            'training': training, 'source_files': [{'path': k, 'sha256': v} for k, v in sorted(sources.items())],
            'generator_sha256': digest(Path(__file__)), 'numpy_version': np.__version__}


def cell(x):
    return f"${num(x['difference_pp'])}\\;[{num(x['ci95_pp'][0])},\\,{num(x['ci95_pp'][1])}]$"


def per_seed(x):
    return '/'.join(f'{v:.2f}' for v in x['seed_mean_differences_pp']).replace('-', '$-$')


def pmv(entry):
    return f"\\pmv{{{entry['mean']:.4f}}}{{{entry['sample_sd']:.4f}}}"


def decoder_table(report):
    s, p, x = report['summaries'], report['planned'], report['exploratory']
    v = FRACTIONS
    lines = [r'% Generated by scripts/analyze_decoder_study_20261105.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Decoder capacity (RQ3; post hoc with respect to the main and common-recipe results, with design '
             r'and planned contrasts fixed before any of these runs). ViT-Large strategies with the common improved recipe '
             r'and either the lightweight decoder with the four-layer readout (0.66 million parameters; the runs of '
             r'Tables~\ref{tab:v2} and~\ref{tab:labels}) or a hierarchical decoder with skip connections from encoder '
             r'blocks 5, 11, 17, and 23 and from the input slice (3.12 million parameters). Subsets, batch order, learning '
             r'rates, checkpoint rule, and evaluation are identical, and all hierarchical-decoder runs used one RTX 5090. '
             r'Dice: mean $\pm$ SD of three seed means. $\Delta$: hierarchical minus lightweight decoder. Differences are '
             r'paired case-bootstrap differences in percentage points with 95\% CIs; per-seed differences follow seed '
             r'order 42/2026/3407.}', r'\label{tab:decoder}',
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xccccc@{}}',
             r'\toprule Strategy & Lightweight & Hierarchical & $\Delta$ [95\% CI] & Per seed & '
             r'\shortstack{\mbox{nnU-Net} 3D $-$\\hierarchical}\\ \midrule']
    for f in (100, 10):
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{v[f]} training volumes}} (\\mbox{{nnU-Net}} 3D: "
                     f"{pmv(s[f'nnU-Net 3D f{f:03d}'])})}}\\\\")
        for st in CELLS[f]:
            d, g = p[f'H - L {st} @{v[f]}'], p[f'nnU-Net 3D - H {st} @{v[f]}']
            lines.append(f"\\quad {STRATEGIES[st]} & {pmv(s[f'L {st} f{f:03d}'])} & {pmv(s[f'H {st} f{f:03d}'])} & "
                         f"{cell(d)} & {per_seed(d)} & {cell(g)}\\\\")
    lines.append(r'\midrule \multicolumn{6}{@{}l}{\textit{Contrasts with the hierarchical decoder, 108 training '
                 r'volumes (unadjusted)}}\\')
    for a, b in STRATEGY_CONTRASTS:
        e = x[f'H {a} - H {b} @108']
        lines.append(f'\\multicolumn{{3}}{{@{{}}l}}{{\\quad {STRATEGIES[a]} $-$ {STRATEGIES[b]}}} & {cell(e)} & '
                     f'{per_seed(e)} & \\\\')
    weak = [(n, e['p_holm']) for n, e in p.items() if e['p_holm'] >= 0.002]
    assert all(n.startswith('H - L ') for n, _ in weak), 'the note below names only decoder contrasts'
    note = (f'$p_{{\\mathrm{{Holm}}}}$ over the {len(p)} planned contrasts: $\\Delta$ for ' +
            ', '.join(f"{STRATEGIES[n.split()[3]]} with {n.split('@')[1]} volumes {fmt_p(q)}" for n, q in weak) +
            '; all others $<0.002$. ')
    # Robustness screen of Section 4.5 and the case x seed bootstrap, as stated in the note.
    clear = [n for n, e in p.items() if e['ci95_pp'][0] > 0 or e['ci95_pp'][1] < 0]
    mixed = [n for n in clear if len({np.sign(y) for y in p[n]['seed_mean_differences_pp']}) > 1]
    joint = [n for n, e in p.items() if e['case_seed_ci95_pp'][0] > 0 or e['case_seed_ci95_pp'][1] < 0]
    assert mixed == ['H - L LO @11'] and sorted(joint) == sorted(set(clear) - set(mixed)), \
        'the robustness statement below would be false'
    screen = ('Every planned contrast whose interval excludes zero has the same sign in all three seeds, except '
              '$\\Delta$ for LoRA with 11 volumes; a case $\\times$ seed bootstrap excludes zero for the other '
              'contrasts with intervals excluding zero and for no further contrast.')
    lines += [r'\bottomrule\end{tabularx}', r'\par\smallskip\raggedright\scriptsize', note + screen, r'\end{table}', '']
    return '\n'.join(lines)


def decoder_seeds_table(report):
    s, t = report['summaries'], report['training']
    lines = [r'% Generated by scripts/analyze_decoder_study_20261105.py; do not edit by hand.',
             r'\begin{table}[!htbp]', r'\centering\small',
             r'\caption{Mean test-volume Dice for each training seed in the decoder study (ViT-Large, common improved '
             r'recipe, hierarchical decoder) and for \mbox{nnU-Net} ResEncM 2D, trained with the recipe of the 3D '
             r'reference on the same 108 training volumes. Selected epoch: epoch of the retained checkpoint for seeds '
             r'42/2026/3407 (DINOv2: highest validation Dice within at most 50 epochs and patience 10; \mbox{nnU-Net}: '
             r'last improvement of the moving-average validation-patch Dice within 1,000 epochs).}',
             r'\label{tab:decoderseeds}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc@{}}', r'\toprule',
             r'Training volumes & Configuration & Seed 42 & Seed 2026 & Seed 3407 & Selected epoch\\ \midrule']
    rows = [(f, f'H {st}', STRATEGIES[st]) for f in (100, 10) for st in CELLS[f]]
    rows.append((100, 'nnU-Net 2D', r'\mbox{nnU-Net} ResEncM 2D'))
    for i, (f, m, label) in enumerate(rows):
        if i and (rows[i - 1][0] != f or m == 'nnU-Net 2D'):
            lines.append(r'\midrule')
        seeds = ' & '.join(f'{y:.4f}' for y in s[f'{m} f{f:03d}']['seed_means'])
        epochs = '/'.join(str(t[f'{m} f{f:03d} seed{seed}']['best_epoch']) for seed in SEEDS)
        lines.append(f'{FRACTIONS[f]} & {label} & {seeds} & {epochs}\\\\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path)
    args = parser.parse_args()
    report = analyze()
    JSON_PATH.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {JSON_PATH}')
    if args.paper_dir:
        for name, text in (('tab_decoder.tex', decoder_table(report)),
                           ('tab_decoder_seeds.tex', decoder_seeds_table(report))):
            (args.paper_dir / name).write_text(text, encoding='utf-8')
            print(f'wrote {args.paper_dir / name}')


if __name__ == '__main__':
    main()
