---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## RNN Exploration

```python
from fastai.text import *
```

```python
bs=64
path=untar_data(URLs.HUMAN_NUMBERS)
path.ls()
```

```python
def readnums(d):
    return [', '.join(o.strip() for o in open(path/d).readlines())]
```

```python
train_text = readnums('train.txt')
train_text[0][:80]
```

```python
valid_text = readnums('valid.txt')
valid_text[0][-80:]
```

```python
train = TextList(train_text, path=path)
valid = TextList(valid_text, path=path)

src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
```

```python
type(src)
```

```python
data = src.databunch(bs=bs)
```

```python
type(data)
```

```python
train[0].text[:80]
```

```python
data.bptt
```

```python
len(data.valid_dl)
```

```python
it = iter(data.valid_dl)
x1,y1 = next(it)
x2,y2 = next(it)
x3,y3 = next(it)
it.close()
```

```python
x1
```

```python
y1
```

```python
x1.numel()
```

```python
x1.shape, y1.shape
```

```python
x2.shape, y2.shape
```

```python
x3.shape, y3.shape
```

```python
v = data.valid_ds.vocab
v.itos
```

```python
doc(v.textify)
```

```python
v.textify(x3[0])
```

```python
data.show_batch(ds_type=DatasetType.Train)
```

```python
data = src.databunch(bs=bs, bptt=3)
x,y = data.one_batch()
x.shape, y.shape
```

```python
nv = len(v.itos)
nv
```

```python
nh = 64
def loss4(input,target): return F.cross_entropy(input, target[:,-1])
def acc4 (input,target): return accuracy(input, target[:,-1])
```

```python
x[:,0]
```

```python
doc(Learner)
```

```python
doc(nn.Module)
```

```python
# Some of the PyTorch items imported by fastai
#
# import torch.nn as nn
# import torch.nn.functional as F
```

```python
class model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Embedding(nv, nh)
        self.hidden_to_hidden = nn.Linear(nh, nh)
        self.hidden_to_output = nn.Linear(nh, nv)
        self.batch_norm = nn.BatchNorm1d(nh)
    
    def forward(self, x):
        h = self.batch_norm(F.relu(self.input_to_hidden(x[:,0])))
        if x.shape[1]>1:
            h = h + self.input_to_hidden(x[:,1])
            h = self.batch_norm(F.relu(self.hidden_to_hidden(h)))
        if x.shape[1]>2:
            h = h + self.input_to_hidden(x[:,2])
            h = self.batch_norm(F.relu(self.hidden_to_hidden(h)))
        return self.hidden_to_output(h)
```

```python
learn = Learner(data, model0(), loss_func=loss4, metrics=acc4)
```

```python
learn.lr_find()
learn.recorder.plot()
```

```python
learn.fit_one_cycle(6, 1e-4)
```

```python
class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Embedding(nv, nh)
        self.hidden_to_hidden = nn.Linear(nh, nh)
        self.hidden_to_output = nn.Linear(nh, nv)
        self.batch_norm = nn.BatchNorm1d(nh)
    
    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.input_to_hidden(x[:,i])
            h = self.batch_norm(F.relu(self.hidden_to_hidden(h)))
        return self.hidden_to_output(h)
```

```python
learn = Learner(data, model1(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)
```

```python
data = src.databunch(bs=bs, bptt=3)
```

```python
x,y = data.one_batch()
x.shape, y.shape
```

```python
doc(nn.Linear)
```

```python
class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Embedding(nv, nh)
        self.hidden_to_hidden = nn.Linear(nh, nh)
        self.hidden_to_output = nn.Linear(nh, nv)
        self.batch_norm = nn.BatchNorm1d(nh)
        
    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.input_to_hidden(x[:,i])
            h = F.relu(self.hidden_to_hidden(h))
            res.append(self.hidden_to_output(self.batch_norm(h)))
        return torch.stack(res, dim=1)
```

```python
learn = Learner(data, model2(), metrics=accuracy)
```

```python
learn.lr_find()
learn.recorder.plot()
```

```python
learn.fit_one_cycle(10, 1e-5, pct_start=0.1)
```

```python
bs
```

```python
class model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Embedding(nv, nh)
        self.hidden_to_hidden = nn.Linear(nh, nh)
        self.hidden_to_output = nn.Linear(nh, nv)
        self.batch_norm = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).to(device=x.device)
        
    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.input_to_hidden(x[:,i])
            h = F.relu(self.hidden_to_hidden(h))
            res.append(self.batch_norm(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.hidden_to_output(res)
        return res
```

```python
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).to(device=x.device)
        
    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res
```

```python
learn = Learner(data, Model3(), metrics=accuracy)
```

```python
learn.lr_find()
learn.recorder.plot()
```

```python
learn.fit_one_cycle(10, 1e-3)
```

```python

```

```python

```
