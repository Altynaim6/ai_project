"""Frozen supervised GCN protocol, using exactly the RF development targets."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.config import PROJECT_ROOT as ROOT
from src.models.gnn import HAIStandaloneGCN
from src.models.baseline import make_features, valid_targets
from src.run_random_forest_hai import transformed
from src.prepare_hai_v2 import audit_source
from src.evaluate_transformer_mixed import calculate_metrics, plot_results, validate_test
from src.train_transformer_mixed import file_hash, write_json, calibrate_scores


def node_features(flat, input_columns, node_order):
    if len(set(node_order)) != len(node_order) or set(input_columns) != set(node_order):
        raise ValueError('Graph channels differ from preprocessing')
    order = [input_columns.index(name) for name in node_order]
    return np.ascontiguousarray(flat.reshape(-1, 6, len(input_columns)).transpose(0, 2, 1)[:, order])


@torch.no_grad()
def scores(model, values, batch_size):
    model.eval()
    return np.concatenate([model(values[i:i+batch_size]).softmax(-1)[:, 1].numpy()
                           for i in range(0, len(values), batch_size)])


def main():
    config = json.loads((ROOT/'configs/gnn_hai.json').read_text())
    torch.set_num_threads(config['threads'])
    schema = json.loads((ROOT/'configs/hai_20_07_schema.json').read_text())
    pre = ROOT/'artifacts/hai/preprocessing_hai_mixed_v2'
    bundle = joblib.load(pre/'preprocessing.joblib')
    metadata = json.loads((pre/'manifest.json').read_text())
    rf = ROOT/'artifacts/hai/random_forest/hai_rf_classifier_v1'
    protocol = json.loads((rf/'protocol.json').read_text())
    if file_hash(pre/'preprocessing.joblib') != protocol['preprocessing_sha256']:
        raise ValueError('RF preprocessing changed')
    graph_dir = ROOT/'artifacts/hai/graphs'/config['graph_version']
    graph = torch.load(graph_dir/'graph.pt', weights_only=True)
    if file_hash(graph_dir/'graph.json') != graph['graph_sha256']:
        raise ValueError('Graph hash mismatch')
    artifact = ROOT/'artifacts/hai/gnn'/config['protocol_version']
    output = ROOT/'results/hai/gnn'/config['protocol_version']
    if artifact.exists() or output.exists():
        raise FileExistsError('Use a new version; frozen experiments are immutable')
    artifact.mkdir(parents=True); output.mkdir(parents=True)
    write_json(artifact/'protocol.json', dict(config, graph_sha256=graph['graph_sha256'],
        preprocessing_sha256=protocol['preprocessing_sha256'],
        training_targets_sha256=file_hash(rf/'training_targets.csv'),
        development_sha256=protocol['development_sha256'], tuning_on_test=False))
    selected = pd.read_csv(rf/'training_targets.csv')
    frames = {}
    for name in ('train1', 'train2'):
        path = ROOT/f'data/raw/hai/{name}.csv'
        if file_hash(path) != protocol['development_sha256'][name]:
            raise ValueError('Development source changed')
        frames[name] = audit_source(path, schema)
    n = len(frames['train2']); first, second = int(.7*n), int(.85*n)
    columns = bundle['numeric_columns']+bundle['binary_columns']
    def features(frame, indices):
        return torch.from_numpy(node_features(make_features(transformed(frame, bundle), indices,
            config['sequence_length']), columns, graph['node_order']))
    train_parts = []
    for name in ('train1', 'train2'):
        picked = selected[selected.source.eq(name)]
        frame = frames[name] if name == 'train1' else frames[name].iloc[:first]
        indices = picked.target_row_index.to_numpy()
        if not np.isin(indices, valid_targets(frame.time, config['sequence_length'])).all():
            raise ValueError('Training window crosses source/gap boundary')
        np.testing.assert_array_equal(frame.time.iloc[indices], picked.timestamp)
        np.testing.assert_array_equal(frame.attack.iloc[indices], picked.label)
        train_parts.append(features(frame, indices))
    train_x = torch.cat(train_parts); train_y = torch.tensor(selected.label.to_numpy(), dtype=torch.long)
    splits = {}
    for name, frame in [('validation', frames['train2'].iloc[first:second].reset_index(drop=True)),
                         ('calibration', frames['train2'].iloc[second:].reset_index(drop=True))]:
        expected = metadata['splits'][name]
        assert len(frame) == expected['rows'] and frame.time.iloc[0] == expected['start'] and frame.time.iloc[-1] == expected['end']
        assert frame.attack.eq(0).all()
        indices = valid_targets(frame.time, config['sequence_length'])
        splits[name] = features(frame, indices)
    batch = config['batch_size']
    for seed in config['seeds']:
        torch.manual_seed(seed); np.random.seed(seed)
        folder = artifact/f'seed{seed}'; folder.mkdir()
        model = HAIStandaloneGCN(graph['edge_index'], graph['node_static'], config['hidden'], config['embedding_dim'], config['dropout'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        counts = torch.bincount(train_y); weights = len(train_y)/(2*counts.float())
        best, stale, history = float('inf'), 0, []
        started = time.perf_counter()
        for epoch in range(1, config['max_epochs']+1):
            model.train(); total = 0.
            for indices in torch.randperm(len(train_x)).split(batch):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_x[indices]), train_y[indices], weight=weights)
                loss.backward(); optimizer.step(); total += loss.item()*len(indices)
            val_scores = scores(model, splits['validation'], batch)
            val_loss = float(-np.log1p(-np.clip(val_scores, 0, 1-1e-7)).mean())
            history.append(dict(epoch=epoch, train_loss=total/len(train_x), validation_loss=val_loss))
            print(f'seed={seed} epoch={epoch} train={total/len(train_x):.6f} normal_val={val_loss:.6f}', flush=True)
            if val_loss < best:
                best, stale = val_loss, 0
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, config=config,
                    graph_sha256=graph['graph_sha256'], preprocessing_sha256=protocol['preprocessing_sha256']), folder/'best_model.pt')
            else:
                stale += 1
            pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
            if stale >= config['patience']: break
        saved = torch.load(folder/'best_model.pt', weights_only=True)
        model.load_state_dict(saved['state_dict'])
        cal_scores = scores(model, splits['calibration'], batch)
        threshold = calibrate_scores(cal_scores, config['threshold_quantile'])
        threshold.pop('validation_windows', None)
        threshold.update(method='normal_calibration_quantile', score='GCN softmax class Attack; not probability-calibrated',
            note='Separate normal calibration split; checkpoint selected on validation, no test tuning')
        threshold.update(calibration_fpr=threshold.pop('validation_fpr'), calibration_windows=len(cal_scores),
            calibration_split='calibration', model_sha256=file_hash(folder/'best_model.pt'),
            graph_sha256=graph['graph_sha256'], preprocessing_sha256=protocol['preprocessing_sha256'],
            best_epoch=saved['epoch'], fit_seconds=time.perf_counter()-started)
        write_json(folder/'threshold.json', threshold)
        np.save(folder/'calibration_scores.npy', cal_scores)
    # Only evaluate after every seed/checkpoint/threshold has been frozen.
    audit = json.loads((ROOT/'manifests/hai_mixed_v2_holdout_audit.json').read_text())
    path = ROOT/audit['holdout']
    assert audit['passed'] and file_hash(path) == audit['extracted_sha256']
    test = pd.read_csv(path, sep=';'); validate_test(test, bundle, metadata, config['sequence_length'])
    indices = valid_targets(test.time, config['sequence_length']); test_x = features(test, indices)
    reference_dir = ROOT/'results/hai/transformer_mixed/mixed_v2_full_seed42_run1/test_evaluation'
    reference = pd.read_csv(reference_dir/'test_predictions.csv')
    np.testing.assert_array_equal(test.time.iloc[indices], reference.timestamp)
    np.testing.assert_array_equal(test.attack.iloc[indices], reference.label)
    rows = []
    for seed in config['seeds']:
        folder = artifact/f'seed{seed}'; result = output/f'seed{seed}'; result.mkdir()
        saved = torch.load(folder/'best_model.pt', weights_only=True)
        threshold = json.loads((folder/'threshold.json').read_text())
        assert file_hash(folder/'best_model.pt') == threshold['model_sha256']
        model = HAIStandaloneGCN(graph['edge_index'], graph['node_static'], config['hidden'], config['embedding_dim'], config['dropout'])
        model.load_state_dict(saved['state_dict'])
        values = scores(model, test_x, batch)
        metrics, predictions = calculate_metrics(test.attack.iloc[indices].to_numpy(), values, threshold['threshold'])
        metrics.update(model='Standalone GCN', seed=seed, best_epoch=saved['epoch'],
            test_previously_seen=True, tuning_on_test=False, test_sha256=audit['extracted_sha256'],
            training_access='normal and labelled attacks; same sampled targets as RF')
        frame = pd.DataFrame(dict(timestamp=test.time.iloc[indices].to_numpy(), label=test.attack.iloc[indices].to_numpy(), score=values, prediction=predictions))
        frame.to_csv(result/'test_predictions.csv', index=False)
        write_json(result/'metrics.json', metrics); plot_results(frame, metrics, result/'figures')
        rows.append(metrics)
        print(f'TEST seed={seed} F1={metrics["f1"]:.4f} recall={metrics["recall"]:.4f} FPR={metrics["fpr"]:.4f}', flush=True)
    for seed in config['seeds']:
        rows.append(json.loads((ROOT/f'results/hai/random_forest/hai_rf_classifier_v1/seed{seed}/metrics.json').read_text()))
    transformer = json.loads((reference_dir/'metrics.json').read_text()); transformer.update(model='Transformer v2', seed=42)
    rows.append(transformer)
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    write_json(output/'status.json', dict(status='completed', test_targets=len(indices), exact_target_alignment=True))


if __name__ == '__main__':
    main()
