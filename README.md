# Benchmarking LSTM vs Transformer on AG News

## Team
| Member | Branch | Task |
|--------|--------|------|
| Maaz | `lstm-branch` | LSTM model implementation |
| Shayan | `transformer-branch` | Transformer model implementation |

## Project Overview
This project benchmarks two deep learning architectures on the AG News 
text classification dataset which has 4 categories: World, Sports, 
Business and Sci/Tech.

- **LSTM** — Bidirectional LSTM with embedding and dropout
- **Transformer** — Encoder-only Transformer with positional encoding

## Dataset
- **Name:** AG News
- **Train samples:** 120,000
- **Test samples:** 7,600
- **Classes:** World, Sports, Business, Sci/Tech

## Project Structure
benchmarking-lstm-vs-transformer-agnews/

├── src/

│   ├── preprocess.py

│   ├── lstm/

│   │   ├── model.py

│   │   └── train.py

│   └── transformer/

│       ├── model.py

│       └── train.py

├── evaluation/

│   └── evaluate.py

├── outputs/

│   ├── lstm_results.json

│   ├── lstm_confusion_matrix.png

│   ├── lstm_curves.png

│   ├── transformer_results.json

│   ├── transformer_confusion_matrix.png

│   ├── transformer_curves.png

│   └── final_comparison.png

├── notebooks/

│   ├── lstm_notebook.ipynb

│   └── transformer_notebook.ipynb

├── main.py

└── requirements.txt 


## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train LSTM
```bash
python main.py --model lstm --epochs 5
```

### Train Transformer
```bash
python main.py --model transformer --epochs 5
```

### Compare Results
```bash
python outputs/compare.py
```

## Results

| Model | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|---------------|--------------|---------------|
| LSTM | 89.34% | 89.23% | 89.23% |
| Transformer | 93.99% | 92.14% | 92.14% |

## Conclusion
Transformer outperforms LSTM on the AG News classification task achieving 
92.14% test accuracy compared to LSTM's 89.23%. The Transformer's 
self-attention mechanism captures global context across the entire sequence 
while LSTM is limited by sequential memory and vanishing gradients over 
long sequences.