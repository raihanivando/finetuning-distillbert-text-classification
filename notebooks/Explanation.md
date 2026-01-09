The notebooks/ folder contains the complete pipeline for fine-tuning DistilBERT on AG News dataset. Each notebook represents a specific stage in the machine learning workflow and should be executed sequentially.

## Notebook Structure
- **01_data_preparation.ipynb**
  Purpose: Load and preprocess the AG News dataset
  What it does:
  - Loads AG News dataset from HuggingFace (sh0416/ag_news)
  - Explores dataset structure and statistics
  - Adjusts labels from 1-4 to 0-3 (PyTorch standard)
  - Combines title and description fields into single text field
  - Splits data into train (90%), validation (10%), and test sets
  - Saves processed datasets to Google Drive
  
  Key Operations:
    ``` bash
    - Load dataset: 120,000 train + 7,600 test samples
    - Label mapping: {0: World, 1: Sports, 2: Business, 3: Sci/Tech}
    - Train/Val/Test split: 108,000 / 12,000 / 7,600
    ```
  Outputs:
  - ag_news_train/ - Training dataset
  - ag_news_val/ - Validation dataset
  - ag_news_test/ - Test dataset

- **02_tokenization.ipynb**
  Purpose: Convert text data into numerical format that BERT understands
  What it does:
  - Loads DistilBERT tokenizer from HuggingFace
  - Tokenizes all text into input IDs and attention masks
  - Adds special tokens: [CLS] (start), [SEP] (end)
  - Applies padding to max length of 128 tokens
  - Converts to PyTorch tensor format
  - Saves tokenized datasets and tokenizer
  
  Key Operations:
    ``` bash
    - Tokenizer: distilbert-base-uncased
    - Max length: 128 tokens
    - Special tokens: [CLS], [SEP], [PAD]
    - Outputs: input_ids, attention_mask, labels
    ```
  Outputs:
  - tokenized_train/ - Tokenized training data
  - tokenized_val/ - Tokenized validation data
  - tokenized_test/ - Tokenized test data
  - tokenizer/ - Saved tokenizer for inference

- **03_model_training.ipynb**
  Purpose: Fine-tune DistilBERT model on AG News dataset
  What it does:
  - Loads pre-trained DistilBERT model (66M parameters)
  - Adds classification head for 4 output classes
  - Configures training parameters (learning rate, batch size, epochs)
  - Trains model using HuggingFace Trainer API
  - Implements AdamW optimizer with learning rate warmup
  - Validates performance after each epoch
  - Saves best model checkpoint based on validation accuracy
  
  Key Operations:
  ``` bash
  - Model: distilbert-base-uncased + classification head
  - Learning rate: 2e-5 with linear warmup (500 steps)
  - Batch size: 16 (train), 32 (eval)
  - Epochs: 3
  - Optimizer: AdamW (weight decay: 0.01)
  - Loss function: Cross-entropy
  ```
  Training Process:
  - Epoch 1: Model learns basic patterns, loss decreases rapidly
  - Epoch 2: Fine-tunes weights, improves accuracy
  - Epoch 3: Achieves optimal performance, prevents overfitting
  Outputs:
  - distilbert_ag_news_final/ - Trained model files
  - pytorch_model.bin - Model weights
  - config.json - Model configuration
  - tokenizer files - For inference
  - results/ - Training checkpoints
  - logs/ - Training logs

- **04_evaluation.ipynb**
  Purpose: Evaluate trained model on test set and analyze performance
  What it does:
  - Loads trained model from Google Drive
  - Runs inference on 7,600 test samples
  - Calculates comprehensive metrics (accuracy, precision, recall, F1)
  - Generates detailed classification report for each category
  - Creates confusion matrix visualization
  - Analyzes misclassified examples
  - Saves predictions and visualizations

  Key Operations:
  ``` bash
  - Metrics: accuracy, precision, recall, F1-score
  - Per-class analysis: performance breakdown by category
  - Confusion matrix: visualize prediction patterns
  - Error analysis: examine misclassified samples
  ```
  Outputs:
  - confusion_matrix.png - Heatmap visualization
  - test_predictions.csv - All predictions with true labels
  - Console output: Detailed classification report
