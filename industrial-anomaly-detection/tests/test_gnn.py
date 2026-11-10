import json
import numpy as np
import torch
from src.config import PROJECT_ROOT
from src.graph_builder import build_graph
from src.models.gnn import HAIStandaloneGCN
from src.run_gnn_hai import node_features


def test_graph_and_independent_batches():
    torch.set_num_threads(2)
    schema = json.loads((PROJECT_ROOT/'configs/hai_20_07_schema.json').read_text())
    graph, edges, static = build_graph(schema)
    assert len(graph['node_order']) == 59
    pairs = set(map(tuple, edges.T.tolist()))
    assert len(pairs) == edges.shape[1]
    assert sum(a == b for a, b in pairs) == 59
    assert all((b, a) in pairs for a, b in pairs)
    model = HAIStandaloneGCN(edges, static, dropout=0).eval()
    values = torch.randn(2, 59, 6)
    torch.testing.assert_close(model(values)[:1], model(values[:1]), atol=1e-6, rtol=1e-5)
    model(values).sum().backward()
    assert model.conv1.lin.weight.grad.abs().sum() > 0
    clone = HAIStandaloneGCN(edges, static, dropout=0).eval()
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(values), clone(values))


def test_node_feature_order():
    flat = np.arange(18, dtype=np.float32).reshape(1, 18)
    actual = node_features(flat, ['a','b','c'], ['c','a','b'])
    np.testing.assert_array_equal(actual[0, 0], flat.reshape(6, 3)[:, 2])
