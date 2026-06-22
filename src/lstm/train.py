import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

def train_model(model, train_loader, test_loader,
                device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    results   = {"train_loss":[], "train_accuracy":[], "val_accuracy":[]}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"[LSTM] Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (torch.argmax(outputs,1)==labels).sum().item()
            total      += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for inp, lbl in test_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                vc += (torch.argmax(model(inp),1)==lbl).sum().item()
                vt += lbl.size(0)
        val_acc = vc/vt

        results["train_loss"].append(total_loss/len(train_loader))
        results["train_accuracy"].append(correct/total)
        results["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Train: {correct/total:.4f} | Val: {val_acc:.4f}")
    return results