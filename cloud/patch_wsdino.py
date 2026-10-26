#!/usr/bin/env python3
"""Patch upstream WS-DINO so the agentcell tools can drive it (idempotent).
- Eval_BBBC021.py: don't read placeholder CSVs at import.
- WS-DINO_BBBC021.py: --channel_index; sampler weights from --data_path and
  --weak_label_header (any #classes); per-channel checkpoint/log names."""
import re, sys
from pathlib import Path
repo = Path(sys.argv[1])

ev = repo / "Eval_BBBC021.py"
s = ev.read_text()
s = s.replace("x0train = pd.read_csv(f'.../BBBC021_annotated.csv')", "x0train = None  # set by caller")
s = s.replace("x0DMSO = pd.read_csv(f'.../BBBC021_DMSO.csv')", "x0DMSO = None  # set by caller")
ev.write_text(s)

tr = repo / "WS-DINO_BBBC021.py"
s = tr.read_text()
if "--channel_index" not in s:
    s = s.replace("    parser.add_argument('--weak_label_header'",
                  "    parser.add_argument('--channel_index', default=0, type=int)\n    parser.add_argument('--weak_label_header'", 1)
# replace module-level weight computation with a function called after args parsed
start = s.index("df = pd.read_csv('.../BBBC021_annotated.csv')")
end = s.index("weighty = 1/torch.tensor(weight_list_flat)") + len("weighty = 1/torch.tensor(weight_list_flat)")
s = s[:start] + '''CHN = ['DAPI', 'Tubulin', 'Actin']


def make_weights(csv_path, header):
    df = pd.read_csv(csv_path)
    counts = df[header].map(df[header].value_counts())
    return 1.0 / torch.tensor(counts.values, dtype=torch.double)
''' + s[end:]
s = s.replace("self.X0 = path0[args.channel_headers[0]]", "self.X0 = path0[args.channel_headers[args.channel_index]]")
s = s.replace("sampler = WeightedRandomSampler(weighty, num_samples=len(weighty))",
              "weighty = make_weights(args.data_path, args.weak_label_header)\n    sampler = WeightedRandomSampler(weighty, num_samples=len(weighty))")
chname = "CHN[args.channel_index]"
s = s.replace("os.path.join(args.output_dir, 'DAPI_weak_compound_DINO_checkpoint.pth')",
              f"os.path.join(args.output_dir, f'{{{chname}}}_weak_compound_DINO_checkpoint.pth')")
s = s.replace("os.path.join(args.output_dir, f'DAPI_weak_compound_DINO_checkpoint{epoch:04}.pth')",
              f"os.path.join(args.output_dir, f'{{{chname}}}_weak_compound_DINO_checkpoint{{epoch:04}}.pth')")
s = s.replace('(Path(args.output_dir) / "DAPI_weak_compound_DINO_log.txt")',
              f'(Path(args.output_dir) / f"{{{chname}}}_weak_compound_DINO_log.txt")')
tr.write_text(s)
print("patched")
