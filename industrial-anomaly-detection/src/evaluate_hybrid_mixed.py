"""Evaluate a completed hybrid run on audited test2 and compare frozen baselines."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.config import PROJECT_ROOT as ROOT
from src.evaluate_transformer_mixed import evaluate, load_frozen_run
from src.train_transformer_mixed import file_hash, write_json


def compare(run_name, batch_size=64, device='cpu'):
    model_dir = ROOT/'artifacts/hai/hybrid_mixed'/run_name
    _, _, calibration, checkpoint = load_frozen_run(model_dir)
    if checkpoint.get('model_kind') != 'hybrid_mixed':
        raise ValueError('Expected hybrid checkpoint')
    if calibration.get('calibration_split') != 'calibration':
        raise ValueError('Hybrid comparison requires separate calibration')
    audit = json.loads((ROOT/'manifests/hai_mixed_v2_holdout_audit.json').read_text())
    test_path = ROOT/audit['holdout']
    if not audit['passed'] or file_hash(test_path) != audit['extracted_sha256']:
        raise ValueError('Holdout audit/hash mismatch')
    reference_root = ROOT/'results/hai/transformer_mixed/mixed_v2_full_seed42_run1/test_evaluation'
    reference = pd.read_csv(reference_root/'test_predictions.csv')
    # Check alignment before scoring or creating an evaluation directory.
    raw = pd.read_csv(test_path, sep=';')
    length = checkpoint['model_config']['sequence_length']
    np.testing.assert_array_equal(raw.time.iloc[length:], reference.timestamp)
    np.testing.assert_array_equal(raw.attack.iloc[length:], reference.label)
    output = ROOT/'results/hai/hybrid_mixed'/run_name/'test_evaluation'
    metrics = evaluate(model_dir, test_path, output, batch_size, device)
    metrics.update(model='Hybrid GCN + Transformer', seed=checkpoint['seed'],
                   training_access='normal only', test_previously_seen=True,
                   descriptive_comparison=True, graph_sha256=checkpoint['model_config']['graph_sha256'])
    write_json(output/'metrics.json', metrics)
    pd.DataFrame([metrics]).to_csv(output/'metrics.csv', index=False)
    rows = [metrics]
    reference_files = [reference_root/'metrics.json']
    for kind, protocol in [('gnn','hai_gcn_classifier_v1'), ('random_forest','hai_rf_classifier_v1')]:
        reference_files += [ROOT/f'results/hai/{kind}/{protocol}/seed{seed}/metrics.json' for seed in [42,43,44]]
    for path in reference_files:
        row = json.loads(path.read_text())
        if row['test_sha256'] != metrics['test_sha256'] or row['evaluated_windows'] != metrics['evaluated_windows']:
            raise ValueError('Comparison source uses different test data/targets')
        if path == reference_files[0]: row.update(model='Transformer v2', seed=42)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'comparison.csv', index=False)
    means = frame.groupby('model')[['precision','recall','f1','fpr','average_precision']].mean()
    means.to_csv(output/'comparison_means.csv')
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, metric in zip(axes,['f1','recall','fpr']):
        ax.bar(means.index,means[metric]); ax.set_title(metric)
        ax.set_ylim(0,1); ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('Retrospective HAI test2 comparison: frozen thresholds, unequal seed counts/label access')
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=150); plt.close(fig)
    (output/'comparison_notes.md').write_text(
        '# Comparison limits\n\nHybrid and Transformer use the same normal-only v2 forecast windows and loss. '
        'RF/GCN see labelled training attacks and current target features. '
        'Baseline RF/GCN means use three seeds; Transformer uses one recovered run; '
        'this hybrid row uses one completed run. Test2 is previously inspected and retrospective. '
        'No threshold is fitted on test. No point adjustment. A new blind claim needs a new protocol/dataset.\n',encoding='utf-8')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('Invalid run name')
    if args.threads < 1: parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    compare(args.run_name,args.batch_size,args.device)


if __name__ == '__main__': main()
