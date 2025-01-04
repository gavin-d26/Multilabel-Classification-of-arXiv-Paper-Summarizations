
# Multilabel Classification of arXiv Paper Summarizations

## Project Description
This project focuses on the multilabel classification of arXiv paper summaries. The goal is to accurately classify the summaries into multiple categories based on their content.

## Installation
To set up the environment and install dependencies, follow these steps:
```bash
git clone https://github.com/gavin-d26/Multilabel-Classification-of-arXiv-Paper-Summarizations.git
cd Multilabel-Classification-of-arXiv-Paper-Summarizations
pip install -r requirements.txt
```

## Usage
To run the project, use the following command:
```bash
python main.py --input data/input_file.csv --output results/output_file.csv
```

## Data Preprocessing
The data preprocessing steps include:
- TF-IDF vectorization with removal of English stop words, 2000 dimensions, ngram_range=(1, 2).
- Experimentation with PCA to reduce dimensions from 2000 to 250, followed by min-max scaling to the [0,1] range.
- Classification on all classes and removing classes with less than 1% occurrences.

## Features
The features used for the best-performing model are as described above but without PCA and min-max scaling.

## Results
The results of the best-performing model are as follows:
- **Model:** Linear SVC
- **Hyperparameters:** {'C': 1, 'penalty': 'l2'}
- **F1 Macro Score:** 0.4052
- **F1 Micro Score:** 0.7718

## Future Work
Future improvements could include further tuning of hyperparameters, experimenting with different models, and enhancing the preprocessing steps.


