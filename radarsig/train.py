from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from radarsig.models import RadarCNN, DenseBaseline


def train_one(model, train_loader, val_loader, device, epochs, lr, augment=False):
    model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if augment:
                # Additional complex Gaussian noise; independent of the stored SNR labels.
                xb = xb + torch.randn_like(xb) * np.random.uniform(0.01, 0.08)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); running += loss.item()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).argmax(1).cpu(); correct += (pred == yb).sum().item(); total += len(yb)
        acc = correct / max(total, 1)
        print(f"epoch {epoch:02d}/{epochs} loss={running/max(len(train_loader),1):.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return best


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--augment',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=Path('artifacts')); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d=np.load(args.data); x,y=d['x'],d['y']; idx=np.arange(len(y))
    tr,va=train_test_split(idx,test_size=.2,random_state=args.seed,stratify=y)
    def loader(ids,shuffle):
        ds=TensorDataset(torch.from_numpy(x[ids]),torch.from_numpy(y[ids]))
        return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=0)
    device='cuda' if torch.cuda.is_available() else 'cpu'; args.output_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for name,model in [('cnn',RadarCNN()),('dense',DenseBaseline())]:
        print(f"\nTraining {name} on {device}; augmentation={args.augment and name=='cnn'}")
        score=train_one(model,loader(tr,True),loader(va,False),device,args.epochs,1e-3,args.augment and name=='cnn')
        torch.save({'state_dict':model.state_dict(),'model':name,'classes':d['class_names'].tolist()},args.output_dir/f'{name}.pt')
        results[name]={'best_validation_accuracy':score}
    (args.output_dir/'training_summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
