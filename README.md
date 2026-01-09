# Fine-tuning BERT for Text Classification on AG News

## Project Overview
This project implements an end-to-end text classification system using DistilBERT, a transformer-based encoder model, fine-tuned on the AG News dataset to categorize news articles into 4 categories: World, Sports, Business, and Sci/Tech.

## Student Information
- **Name**: [Muhamad Mario Rizki],[Raihan Ivando Diaz],[Abid Sabyano Rozhan]
- **Student ID**: [1103223063],[1103223093],[1103220222]
- **Course**: Deep Learning - Final Term
- **Task**: Task 1 - Encoder/BERT-family for Classification

## Model Architecture
- **Model**: DistilBERT (distilbert-base-uncased)
- **Architecture Type**: Encoder-only Transformer
- **Parameters**: ~66 million
- **Framework**: HuggingFace Transformers, PyTorch

## Dataset
- **Name**: AG News
- **Source**: `sh0416/ag_news` from HuggingFace Datasets
- **Classes**: 4 (World, Sports, Business, Sci/Tech)
- **Training samples**: 108,000
- **Validation samples**: 12,000
- **Test samples**: 7,600

## Project Structure
```bash
finetuning-bert-text-classification/
├── README.md # Project documentation
├── notebooks/ # Jupyter notebooks
│ ├── 01_data_preparation.ipynb # Data loading and preprocessing
│ ├── 02_tokenization.ipynb # Text tokenization
│ ├── 03_model_training.ipynb # Model fine-tuning
│ ├── 04_evaluation.ipynb # Model evaluation
│ └── 05_inference_demo.ipynb # Usage demo (optional)
├── reports/ # Results and visualizations
│ ├── confusion_matrix.png # Confusion matrix
│ ├── training_results.md # Training metrics
│ └── analysis.md # Result analysis
└── requirements.txt # Python dependencies
```

## Results
**Model Performances**
- Accuracy: 94.53%
- Precision: 0.9454
- Recall: 0.9453
- F1-Score: 0.9453

## Per Class Performance
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| World    | 0.9581    | 0.9500 | 0.9540   | 1900    |
| Sports   | 0.9863    | 0.9879 | 0.9871   | 1900    |
| Business | 0.9264    | 0.9079 | 0.9171   | 1900    |
| Sci/Tech | 0.9108    | 0.9353 | 0.9229   | 1900    |

**Key Findings**
- Sports category achieved the highest accuracy (98.71% F1-score) due to distinctive terminology
- Business category showed the most confusion with other categories (91.71% F1-score)
- Overall performance exceeds 94%, indicating strong generalization

**Training Configuration**
- Learning Rate: 2e-5
- Batch Size (train): 16
- Batch Size (eval): 32
- Epochs: 3
- Optimizer: AdamW with weight decay (0.01)
- Warmup Steps: 500
- Max Sequence Length: 128 tokens
