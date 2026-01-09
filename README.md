# Fine-tuning BERT for Text Classification on AG News

## Project Overview
This project implements an end-to-end text classification system using DistilBERT, a transformer-based encoder model, fine-tuned on the AG News dataset to categorize news articles into 4 categories: World, Sports, Business, and Sci/Tech.

## Student Information
- **Name**: [Muhamad Mario Rizki],[Raihan Ivando Diaz],[Abid Sabyano Rozhan]
- **Student ID**: [1103223063],[110322xxxx],[110322xxxx]
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

## 
