import torch
from radarsig.data.generate import make_dataset, CLASS_NAMES
from radarsig.models import RadarCNN, DenseBaseline

def test_generator_shapes_and_labels():
    x,y,s=make_dataset(samples=32,seed=1,length=64)
    assert x.shape==(32,2,64) and y.shape==(32,) and s.shape==(32,)
    assert set(y.tolist())==set(range(len(CLASS_NAMES)))

def test_models_forward():
    x=torch.randn(4,2,256)
    assert RadarCNN()(x).shape==(4,8)
    assert DenseBaseline()(x).shape==(4,8)
