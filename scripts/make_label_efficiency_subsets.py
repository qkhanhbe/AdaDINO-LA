"""Write the pre-specified nested training subsets of the label-efficiency study (no training)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.label_efficiency_study import FRACTIONS, PROTOCOL, SUBSET_STREAM, all_subsets  # noqa: E402
from src.train_e5_e6 import audit_protocol_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/utah2018_run2")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/label_efficiency_subsets.json")
    args = parser.parse_args()
    ids, _ = audit_protocol_data(args.data_root)
    subsets = all_subsets(ids["train"])
    value = {"study": PROTOCOL["name"], "status": PROTOCOL["status"], "subset_rule": PROTOCOL["subset_rule"],
             "subset_stream": SUBSET_STREAM, "fractions_percent_to_volumes": FRACTIONS,
             "validation_volumes": len(ids["val"]), "test_volumes": len(ids["test"]), "subsets": subsets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
