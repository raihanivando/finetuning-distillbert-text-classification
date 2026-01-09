# Training Results

## Training Configuration
- Model: DistilBERT (distilbert-base-uncased)
- Dataset: AG News (sh0416/ag_news)
- Training Samples: 108,000
- Validation Samples: 12,000
- Test Samples: 7,600

## Hyperparameters
- Learning Rate: 2e-5
- Batch Size (train): 16
- Batch Size (eval): 32
- Epochs: 3
- Max Length: 128 tokens
- Optimizer: AdamW
- Weight Decay: 0.01
- Warmup Steps: 500

## Test Set Performance

### Overall Metrics
- **Accuracy**: 94.53%
- **Precision**: 0.9454
- **Recall**: 0.9453
- **F1-Score**: 0.9453

### Per-Class Metrics
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| World    | 0.9581    | 0.9500 | 0.9540   | 1900    |
| Sports   | 0.9863    | 0.9879 | 0.9871   | 1900    |
| Business | 0.9264    | 0.9079 | 0.9171   | 1900    |
| Sci/Tech | 0.9108    | 0.9353 | 0.9229   | 1900    |

### Training Time
- Total Training Time: ~1 hour
- Hardware: Google Colab GPU (T4)
