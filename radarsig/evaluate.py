from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from radarsig.models import RadarCNN


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,default=Path('artifacts/cnn.pt'))
    ap.add_argument('--output-dir',type=Path,default=Path('artifacts/evaluation')); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    d=np.load(args.data); x,y,snr=d['x'],d['y'],d['snr']; classes=d['class_names'].tolist()
    _,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=args.seed,stratify=y)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model=RadarCNN(len(classes)); model.load_state_dict(ck['state_dict']); model.eval()
    loader=DataLoader(TensorDataset(torch.from_numpy(x[te]),torch.from_numpy(y[te])),batch_size=512)
    preds=[]
    with torch.no_grad():
        for xb,_ in loader: preds.extend(model(xb).argmax(1).numpy().tolist())
    preds=np.asarray(preds); truth=y[te]; args.output_dir.mkdir(parents=True,exist_ok=True)
    print(f"Held-out accuracy: {accuracy_score(truth,preds):.4f}")
    print(classification_report(truth,preds,target_names=classes,zero_division=0))
    rows=[]
    for level in sorted(np.unique(snr[te])):
        mask=snr[te]==level; acc=accuracy_score(truth[mask],preds[mask]); rows.append((int(level),int(mask.sum()),float(acc)))
        cm=confusion_matrix(truth[mask],preds[mask],labels=np.arange(len(classes)))
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
        ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,yticklabels=classes,
               xlabel='Predicted',ylabel='True',title=f'Confusion matrix at {level} dB')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right'); fig.tight_layout(); fig.savefig(args.output_dir/f'confusion_{level:+03d}dB.png'); plt.close(fig)
    with (args.output_dir/'accuracy_by_snr.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['snr_db','examples','accuracy']); w.writerows(rows)
    print(f"Saved per-SNR report to {args.output_dir}")
if __name__=='__main__': main()
