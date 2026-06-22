import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("outputs/lstm_results.json") as f:
    lstm = json.load(f)

with open("outputs/transformer_results.json") as f:
    trans = json.load(f)

epochs = range(1, len(lstm["val_accuracy"]) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, lstm["val_accuracy"],  marker="o", label="LSTM",        color="green")
plt.plot(epochs, trans["val_accuracy"], marker="s", label="Transformer",  color="purple")
plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs, lstm["train_loss"],  marker="o", label="LSTM",       color="green")
plt.plot(epochs, trans["train_loss"], marker="s", label="Transformer", color="purple")
plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs, lstm["train_accuracy"],  marker="o", label="LSTM",       color="green")
plt.plot(epochs, trans["train_accuracy"], marker="s", label="Transformer", color="purple")
plt.title("Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/final_comparison.png", dpi=100, bbox_inches='tight')
plt.close()

print("\n========== FINAL RESULTS ==========")
print(f"LSTM        — Val Accuracy:  {lstm['val_accuracy'][-1]:.4f}")
print(f"Transformer — Val Accuracy:  {trans['val_accuracy'][-1]:.4f}")
print(f"LSTM        — Test Accuracy: {lstm['final_test_accuracy']:.4f}")
print(f"Transformer — Test Accuracy: {trans['final_test_accuracy']:.4f}")
print("====================================")
winner = "Transformer" if trans["final_test_accuracy"] > lstm["final_test_accuracy"] else "LSTM"
print(f"\nWinner: {winner}")
print("Saved outputs/final_comparison.png")