"""Post hoc sensitivity of the CDEMRIS comparison with nnU-Net (6/11/2026).

In one CDEMRIS volume (pre_p10) nnU-Net failed for all seeds, and in another (post_p18, an oblique isotropic
slab) every model predicted no LA. This script repeats the per-configuration differences of Table external
(configuration minus nnU-Net, patient-level bootstrap) without pre_p10, and without both volumes. It reuses the
inputs, configuration set and bootstrap of scripts/analyze_revision_v2_20261021.py and first checks that the
40-volume values reproduce outputs/revision_v2.json. Defined after the external results were
viewed, so it is exploratory. The common-recipe rows (external/dino_v2, scored post hoc) are included when present
and summarized separately from the main-protocol rows.

Run: python -B scripts/analyze_external_sensitivity_20261106.py
Writes outputs/external_sensitivity.json.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_revision_v2 import (EXT_CONFIGS, EXT_FP32, EXT_V2, REV, SEEDS,  # noqa: E402
                                          analyze_external, cluster_bootstrap, digest)

REFERENCE = ROOT / 'outputs/revision_v2.json'
JSON_PATH = ROOT / 'outputs/external_sensitivity.json'
SUBSETS = {'cdemris40': (), 'without_pre_p10': ('pre_p10',), 'without_pre_p10_post_p18': ('pre_p10', 'post_p18')}


def main():
    sources = {}

    def track(path):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return path

    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))['external']
    recomputed = analyze_external(track)
    v2_present = (REV / 'external/dino_v2').is_dir()
    groups = {'main_protocol': list(EXT_CONFIGS), 'common_recipe': list(EXT_V2) if v2_present else []}
    configs = groups['main_protocol'] + groups['common_recipe']
    for k in configs:  # the 40-volume differences must match the published table inputs exactly
        assert recomputed[k]['cdemris40']['minus_nnunet'] == reference[k]['cdemris40']['minus_nnunet'], k

    # Per-case seed means from the same files and configuration set as the table.
    def load(folder):
        out = {}
        for seed in SEEDS:
            with track(folder / f'seed{seed}.csv').open(encoding='utf-8-sig', newline='') as stream:
                for r in csv.DictReader(stream):
                    if r['dataset'].startswith('cdemris'):
                        out.setdefault(r['case'], {})[seed] = float(r['dice_3d'])
        return {c: float(np.mean([v[s] for s in SEEDS])) for c, v in out.items()}

    data = {k: load(REV / ('external/dino_fp32' if k in EXT_FP32 else 'external/dino_original') / k) for k in EXT_CONFIGS}
    if v2_present:
        data.update({k: load(REV / 'external/dino_v2' / v) for k, v in EXT_V2.items()})
    nn = load(REV / 'external/nnunet')
    cases = sorted(nn)
    assert len(cases) == 40 and all(sorted(data[k]) == cases for k in configs)

    result = {}
    for name, dropped in SUBSETS.items():
        ks = [c for c in cases if c not in dropped]
        entry = {'n_volumes': len(ks), 'nnunet_mean': float(np.mean([nn[c] for c in ks])), 'configurations': {}}
        for k in configs:
            d = np.array([data[k][c] - nn[c] for c in ks])
            entry['configurations'][k] = {'mean': float(np.mean([data[k][c] for c in ks])),
                                          'minus_nnunet': cluster_bootstrap(d, [c.split('_', 1)[1] for c in ks])}
        for group, members in groups.items():
            if not members:
                continue
            fx = [entry['configurations'][k]['minus_nnunet'] for k in members]
            entry[group] = {'range_pp': [min(x['difference_pp'] for x in fx), max(x['difference_pp'] for x in fx)],
                            'median_range_pp': [min(x['median_pp'] for x in fx), max(x['median_pp'] for x in fx)],
                            'all_intervals_include_zero': all(x['ci95_pp'][0] <= 0 <= x['ci95_pp'][1] for x in fx)}
        entry['range_pp'] = entry['main_protocol']['range_pp']  # key of the first version (main protocol only)
        entry['all_intervals_include_zero'] = all(  # every configuration, both groups
            v['minus_nnunet']['ci95_pp'][0] <= 0 <= v['minus_nnunet']['ci95_pp'][1] for v in entry['configurations'].values())
        result[name] = entry
    # Seed-averaged Dice of the two volumes, for the text.
    result['volumes'] = {c: {'nnU-Net': nn[c], **{k: data[k][c] for k in configs}} for c in ('pre_p10', 'post_p18')}
    report = {'status': 'post hoc, defined after the external results were viewed', 'subsets': result,
              'source_files': [{'path': k, 'sha256': v} for k, v in sorted(sources.items())],
              'generator_sha256': digest(Path(__file__)), 'numpy_version': np.__version__}
    JSON_PATH.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {JSON_PATH}')


if __name__ == '__main__':
    main()
