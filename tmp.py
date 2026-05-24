import csv
from pathlib import Path
from collections import defaultdict

root = Path('/home/nicholas/Desktop/UEM-likelihood-ratio')
in_csv = root / 'results' / 'ood_metrics_table.csv'
out_md = root / 'results' / 'ood_model_comparisons.md'
out_csv = root / 'results' / 'ood_model_comparisons.csv'

rows = []
with in_csv.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'folder': r['folder'],
            'method': r['method'],
            'class': r['class'],
            'AUROC': float(r['AUROC']),
            'AUPRC': float(r['AUPRC']),
            'FPR@TPR95': float(r['FPR@TPR95']),
        })

# index by folder/method/class
idx = {(r['folder'], r['method'], r['class']): r for r in rows}
folders = sorted({r['folder'] for r in rows})

# pair base with base_new where both exist
print(folders)
pairs = []
for f in folders:
    if f.endswith('_new'):
        continue
    new_f = f + '_new'
    if new_f in folders:
        pairs.append((f, new_f))

print(pairs)

metrics = ['AUROC', 'AUPRC', 'FPR@TPR95']
methods = ['OOD', 'LLR']
class_order = ['all', 'all_without_potholes', 'tiny', 'small', 'medium', 'large']

comp_rows = []
for base, new in pairs:
    classes = sorted({r['class'] for r in rows if r['folder'] in (base, new)}, key=lambda c: (class_order.index(c) if c in class_order else 999, c))
    for cls in classes:
        for method in methods:
            a = idx.get((base, method, cls))
            b = idx.get((new, method, cls))
            if not a or not b:
                continue
            rec = {
                'model': base.replace('d_d_', ''),
                'baseline': base,
                'new': new,
                'method': method,
                'class': cls,
            }
            for m in metrics:
                rec[f'{m}_baseline'] = a[m]
                rec[f'{m}_new'] = b[m]
                rec[f'{m}_delta'] = b[m] - a[m]
            comp_rows.append(rec)

# write csv
fieldnames = [
    'model','baseline','new','method','class',
    'AUROC_baseline','AUROC_new','AUROC_delta',
    'AUPRC_baseline','AUPRC_new','AUPRC_delta',
    'FPR@TPR95_baseline','FPR@TPR95_new','FPR@TPR95_delta'
]
with out_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(comp_rows)

# write markdown grouped per model pair
with out_md.open('w') as f:
    f.write('# Baseline vs New Model Comparisons\n\n')
    f.write('Delta columns are computed as `new - baseline` (for FPR, negative is better).\n\n')
    for base, new in pairs:
        model = base.replace('d_d_', '')
        f.write(f'## {model}: `{base}` vs `{new}`\n\n')
        f.write('| Method | Class | AUROC (base) | AUROC (new) | Delta | AUPRC (base) | AUPRC (new) | Delta | FPR (base) | FPR (new) | Delta |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        classes = sorted({r['class'] for r in rows if r['folder'] in (base, new)}, key=lambda c: (class_order.index(c) if c in class_order else 999, c))
        for cls in classes:
            for method in methods:
                a = idx.get((base, method, cls))
                b = idx.get((new, method, cls))
                if not a or not b:
                    continue
                f.write(
                    f"| {method} | {cls} | {a['AUROC']:.6f} | {b['AUROC']:.6f} | {b['AUROC']-a['AUROC']:+.6f} | "
                    f"{a['AUPRC']:.6f} | {b['AUPRC']:.6f} | {b['AUPRC']-a['AUPRC']:+.6f} | "
                    f"{a['FPR@TPR95']:.6f} | {b['FPR@TPR95']:.6f} | {b['FPR@TPR95']-a['FPR@TPR95']:+.6f} |\n"
                )
        f.write('\n')

print(f'Pairs: {len(pairs)}')
print(f'Rows: {len(comp_rows)}')
print(out_md)
print(out_csv)
