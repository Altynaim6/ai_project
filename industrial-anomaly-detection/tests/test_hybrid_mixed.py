import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.config import PROJECT_ROOT as ROOT
from src.graph_builder import build_graph
from src.models.hybrid_mixed import HybridMixedForecast
from src.models.forecast_factory import create_forecast_model
from src.preprocess_hai_mixed import prepare
from src.train_transformer_mixed import run_training
from src.evaluate_transformer_mixed import load_frozen_run, evaluate


def model_config():
    schema=json.loads((ROOT/'configs/hai_20_07_schema.json').read_text())
    graph,edges,static=build_graph(schema)
    inputs=schema['numeric_columns']+schema['binary_columns']
    return dict(edge_index=edges.tolist(),node_static=static.tolist(),
        graph_to_input=[inputs.index(n) for n in graph['node_order']],node_order=graph['node_order'],
        graph_sha256='a'*64,num_numeric=55,num_binary=4,sequence_length=4,
        graph_hidden=4,embedding_dim=2,d_model=8,nhead=2,num_layers=1,dropout=0.)


def test_hybrid_causality_gradients_and_batch_independence():
    torch.set_num_threads(2); torch.manual_seed(42)
    model=HybridMixedForecast(**model_config()).eval()
    history=torch.randn(2,4,59)
    numeric,binary=model(history)
    assert numeric.shape==(2,55) and binary.shape==(2,4,2)
    before=model.encode(history)
    changed=history.clone(); changed[:,2:]+=50
    torch.testing.assert_close(before[:,:2],model.encode(changed)[:,:2])
    torch.testing.assert_close(numeric[:1],model(history[:1])[0],atol=1e-6,rtol=1e-5)
    (numeric.square().mean()+binary.square().mean()).backward()
    assert model.spatial1.weight.grad.abs().sum()>0
    assert model.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    restored=create_forecast_model('hybrid_mixed',model.model_config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(history)[0],restored(history)[0])


def test_graph_changes_spatial_messages_and_invalid_mapping_rejected():
    config=model_config(); model=HybridMixedForecast(**config).eval()
    values=torch.randn(1,4,59)
    original=model.spatial_encode(values)
    with torch.no_grad(): model.normalized_adjacency.copy_(torch.eye(59))
    assert not torch.allclose(original,model.spatial_encode(values))
    config['graph_to_input']=[0]*59
    with pytest.raises(ValueError,match='permutation'): HybridMixedForecast(**config)


def test_full_hybrid_training_calibration_reload_and_evaluation(tmp_path):
    torch.set_num_threads(2)
    schema=json.loads((ROOT/'configs/hai_20_07_schema.json').read_text())
    for split,offset in [('train',0),('validation',40)]:
        frame=pd.DataFrame({'time':pd.date_range('2020-01-01',periods=80,freq='s')[offset:offset+40].astype(str)})
        for name in schema['feature_order']:
            frame[name]=np.arange(40)%2 if name in schema['binary_columns'] else np.sin(np.arange(40))
        for name in ['attack','attack_P1','attack_P2','attack_P3']: frame[name]=0
        frame.to_csv(tmp_path/f'{split}.csv',sep=';',index=False)
    prepare(tmp_path/'train.csv',tmp_path/'validation.csv',ROOT/'configs/hai_20_07_schema.json',tmp_path/'data',tmp_path/'pre')
    with np.load(tmp_path/'data/validation.npz') as data:
        values={key:data[key].copy() for key in data.files}
    values['timestamps']=pd.date_range('2020-01-02',periods=40,freq='s').astype(str).to_numpy(dtype=str)
    np.savez(tmp_path/'data/calibration.npz',**values)
    run_training(tmp_path/'data',tmp_path/'pre',tmp_path/'model',tmp_path/'results',
        dict(max_epochs=1,early_stopping_patience=1,batch_size=8,validation_batch_size=7,learning_rate=.001),
        model_config(),progress_every=0,model_kind='hybrid_mixed')
    model,_,threshold,_=load_frozen_run(tmp_path/'model')
    assert isinstance(model,HybridMixedForecast)
    assert threshold['calibration_split']=='calibration'
    scores=pd.read_csv(tmp_path/'results/calibration_scores.csv').score
    assert threshold['threshold']==pytest.approx(float(np.quantile(scores,.99)))
    frozen=(tmp_path/'model/threshold.json').read_bytes()
    frame['time']=pd.date_range('2020-02-01',periods=40,freq='s').astype(str)
    frame.loc[10:15,'attack']=1
    frame.to_csv(tmp_path/'test.csv',sep=';',index=False)
    metrics=evaluate(tmp_path/'model',tmp_path/'test.csv',tmp_path/'evaluation',batch_size=7)
    assert metrics['evaluated_windows']==36 and metrics['attack_windows']==6
    assert (tmp_path/'model/threshold.json').read_bytes()==frozen
    config=json.loads((tmp_path/'model/run_config.json').read_text())
    config['model_config']['graph_to_input'][0]=(config['model_config']['graph_to_input'][0]+1)%59
    (tmp_path/'model/run_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration differs'): load_frozen_run(tmp_path/'model')
