"""Train a normal-only GCN + causal Transformer using frozen HAI v2 splits."""
import argparse
import json
import joblib
import torch
from src.config import PROJECT_ROOT as ROOT
from src.train_transformer_mixed import run_training, file_hash


def graph_model_config(config, graph_dir, bundle):
    graph = torch.load(graph_dir/'graph.pt', weights_only=True, map_location='cpu')
    digest = file_hash(graph_dir/'graph.json')
    metadata = json.loads((graph_dir/'graph.json').read_text())
    if digest != graph['graph_sha256'] or metadata['node_order'] != graph['node_order']:
        raise ValueError('Graph artifact hash/order mismatch')
    from src.graph_builder import build_graph
    schema = json.loads((ROOT/'configs/hai_20_07_schema.json').read_text())
    rebuilt, edges, static = build_graph(schema)
    if rebuilt['node_order'] != graph['node_order'] or not torch.equal(edges, graph['edge_index']) or not torch.equal(static, graph['node_static']):
        raise ValueError('Graph tensor differs from documented graph builder')
    columns = bundle['numeric_columns']+bundle['binary_columns']
    if set(columns) != set(graph['node_order']) or len(columns) != len(graph['node_order']):
        raise ValueError('Graph/preprocessing channels differ')
    return dict(config, edge_index=edges.tolist(), node_static=static.tolist(),
                graph_to_input=[columns.index(n) for n in graph['node_order']],
                node_order=graph['node_order'], graph_sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--max-train-batches', type=int, help='Smoke only; full validation/calibration still run')
    args = parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('Invalid run name')
    if args.threads < 1: parser.error('threads must be positive')
    config = json.loads((ROOT/'configs/hybrid_hai.json').read_text())
    preprocessing = ROOT/'artifacts/hai/preprocessing_hai_mixed_v2'
    if not (ROOT/'data/processed/hai_mixed_v2/calibration.npz').is_file():
        raise ValueError('Separate v2 calibration split is required')
    model_config = graph_model_config(config['model'], ROOT/'artifacts/hai/graphs'/config['graph_version'],
                                      joblib.load(preprocessing/'preprocessing.joblib'))
    training = dict(config['training'])
    training['threshold_quantile'] = config['threshold_quantile']
    if args.epochs is not None: training['max_epochs'] = args.epochs
    if args.batch_size is not None: training['batch_size'] = args.batch_size
    torch.set_num_threads(args.threads)
    run_training(ROOT/'data/processed/hai_mixed_v2', preprocessing,
        ROOT/'artifacts/hai/hybrid_mixed'/args.run_name,
        ROOT/'results/hai/hybrid_mixed'/args.run_name, training, model_config,
        seed=args.seed, device=args.device, max_train_batches=args.max_train_batches,
        model_kind='hybrid_mixed')


if __name__ == '__main__': main()
