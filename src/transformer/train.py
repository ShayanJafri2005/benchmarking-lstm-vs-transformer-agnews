import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_model(model, train_loader, test_loader,
                device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    results   = {"train_loss":[], "train_accuracy":[], "val_accuracy":[]}
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader,
                desc=f"[Transformer] Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (torch.argmax(outputs,1)==labels).sum().item()
            total      += labels.size(0)
        scheduler.step()
        val_acc = _eval(model, test_loader, device)
        results["train_loss"].append(total_loss/len(train_loader))
        results["train_accuracy"].append(correct/total)
        results["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}"
              f" | Train: {correct/total:.4f} | Val: {val_acc:.4f}")
    return results

def _eval(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (torch.argmax(model(inputs),1)==labels).sum().item()
            total   += labels.size(0)
    model.train()
    return correct / total