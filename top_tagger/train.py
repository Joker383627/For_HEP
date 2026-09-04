
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import numpy as np

from sklearn.metrics import roc_auc_score,roc_curve

@torch.no_grad()
def evaluate(model:nn.Module , loader : DataLoader):
    model.eval()
    all_labels,all_scores = [],[]
    loss_sum = 0.0
    num_sample = 0.0
    for batch in loader:
        x = batch["features"]
        p4 = batch["4-momentum"]
        mask = batch["Mask"]
        y = batch["Jet Label"]

        logits = model(x,mask,p4)

        num_sample += y.size(0)

        loss_sum += F.cross_entropy(logits,y.long(),reduction="sum").item()
        probs = F.softmax(logits,dim = -1)[:,1]

        all_labels.append(y)
        all_scores.append(probs)

    avg_loss = loss_sum/num_sample
    Y = torch.cat(all_labels)
    P_top = torch.cat(all_scores)
    prediction = (P_top>0.5).long()
    accuracy = (prediction == Y).numpy().astype(int).mean()

    return {"loss" : avg_loss, 
            "accuracy" : accuracy , 
            "Y" : Y , 
            "probability_top" : P_top,
            "AUC Score" : roc_auc_score(Y.numpy(),P_top.numpy())}



def train(model:nn.Module, loader:DataLoader, epochs = 10, lr=1e-3, weight_decay = 1e-2):
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr,weight_decay=weight_decay)
    hist = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_AUC Score": []}  
    for epoch in range(epochs):
        loss_sum, n_batch = 0.0,0.0
        model.train()
        for batch in loader["train"]:
            x = batch["features"]
            p4 = batch["4-momentum"]
            mask = batch["Mask"]
            y = batch["Jet Label"]

            logits = model(x,mask,p4)
            loss = F.cross_entropy(logits,y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batch += 1

        epoch_loss = loss_sum/max(1,n_batch)
        hist["train_loss"].append(epoch_loss)

        validation = evaluate(model,loader["val"])

        for keys in ("loss","accuracy","AUC Score"):
            hist["val_"+keys].append(validation[keys])

        print(f"epoch {epoch+1:2d}: train loss {epoch_loss:.3f} | val loss {validation['loss']:.3f} | "
                      f"acc {validation['accuracy']:.3f} | AUC {validation['AUC Score']:.4f}")
    model.history = hist
    return model

        
def background_rejection(Y, P_top, signal_eff=0.5):
    FPR, TPR, theshold = roc_curve(Y,P_top)
    eps_b = np.interp(signal_eff, TPR, FPR)
    N_background = max(int((Y == 0).sum()),1)
    eps_b = max(eps_b, 1.0 /N_background)  
    return 1.0 / eps_b