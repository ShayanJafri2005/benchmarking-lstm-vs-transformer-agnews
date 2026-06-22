import torch, argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src.preprocess import load_agnews
    train_loader, test_loader, vocab = load_agnews(args.batch_size)

    if args.model == "lstm":
        from src.lstm.model import LSTMClassifier
        from src.lstm.train import train_model
        model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    else:
        from src.transformer.model import TransformerClassifier
        from src.transformer.train import train_model
        model = TransformerClassifier(vocab_size=len(vocab)).to(device)

    from evaluation.evaluate import evaluate_model
    results = train_model(model, train_loader, test_loader, device, args.epochs, args.lr)
    acc = evaluate_model(model, test_loader, device, model_name=args.model.upper())

    os.makedirs("outputs", exist_ok=True)
    results["final_test_accuracy"] = acc
    with open(f"outputs/{args.model}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to outputs/{args.model}_results.json")

if __name__ == "__main__":
    main()