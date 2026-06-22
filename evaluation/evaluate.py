import torch, os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def evaluate_model(model, test_loader, device, model_name="Model"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    print(f"\n{'='*50}\n{model_name} — Accuracy: {acc:.4f}\n{'='*50}")
    print(classification_report(targets, preds, target_names=LABELS))

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS,
                yticklabels=LABELS, cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{model_name.lower()}_confusion_matrix.png")
    plt.close()
    return acc