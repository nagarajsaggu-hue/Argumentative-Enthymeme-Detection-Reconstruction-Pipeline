<p align="center">
  <img src="edr.png" alt="Project Repository Banner" width="100%">
</p>


# Argumentative Enthymeme Detection & Reconstruction Pipeline

## Overview

This repository presents an end-to-end Natural Language Processing (NLP) pipeline for **Argumentative Enthymeme Detection and Reconstruction**. The project focuses on identifying missing argumentative components (enthymemes) within arguments and reconstructing the omitted content using state-of-the-art Transformer models.

The system is divided into two major tasks:

1. **Enthymeme Detection** – Identifying the correct position of a missing argumentative statement.
2. **Enthymeme Reconstruction** – Generating the missing argumentative content once the position is known.

The pipeline leverages modern deep learning architectures, including **DeBERTa** for detection and **BART** for text reconstruction.

---

## Project Motivation

Enthymemes are incomplete arguments where one or more premises or conclusions are omitted but implicitly understood by readers.

Automatically detecting and reconstructing these missing components is important for:

* Argument Mining
* Computational Argumentation
* Educational Technology
* Essay Analysis
* Explainable Artificial Intelligence (XAI)
* Natural Language Understanding

This project aims to build a complete workflow capable of locating and reconstructing missing argumentative information.

---

## Project Architecture

```text
Original Argument
        │
        ▼
Dataset Construction
        │
        ▼
Candidate Gap Generation
        │
        ▼
DeBERTa-Based Detection Model
        │
        ▼
Best Gap Position Selection
        │
        ▼
[MASK] Insertion
        │
        ▼
BART Reconstruction Model
        │
        ▼
Generated Missing Statement
```

---

## Features

### Enthymeme Detection

* Candidate gap generation
* Positive and negative sample creation
* Transformer-based classification
* DeBERTa sentence-pair modeling
* Threshold optimization
* Best-position selection

### Enthymeme Reconstruction

* Mask-based text infilling
* BART sequence-to-sequence generation
* Missing statement generation
* Evaluation using standard text-generation metrics

### Evaluation

Detection Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1

Reconstruction Metrics:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* BERTScore

---

## Models Used

### Detection Model

**Microsoft DeBERTa Base**

* Architecture: Transformer Encoder
* Input Format:

```text
Text Left [SEP] Text Right
```

* Loss Function:

  * Weighted Cross Entropy Loss

### Reconstruction Model

**Facebook BART Large**

* Architecture:

  * Encoder-Decoder Transformer

* Input Format:

```text
Argument with [MASK]
```

* Output:

```text
Generated Missing Statement
```

---

## Dataset Format

### Detection Dataset

```csv
id,input_text,label
CNHK1158,text before gap [SEP] text after gap,1
CNHK1158,incorrect gap placement example,0
```

### Reconstruction Dataset

```csv
id,input_text,target_text
CNHK1158,Argument containing [MASK],Missing Statement
```

---

## Repository Structure

```text
Argumentative-Enthymeme-Detection-Reconstruction-Pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reconstruction/
│
├── scripts/
│   ├── preprocessing/
│   ├── detection/
│   ├── reconstruction/
│   └── evaluation/
│
├── models/
│   ├── deberta_detection/
│   └── bart_reconstruction/
│
├── results/
│   ├── detection/
│   └── reconstruction/
│
├── notebooks/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/nagarajsaggu-hue/Argumentative-Enthymeme-Detection-Reconstruction-Pipeline.git

cd Argumentative-Enthymeme-Detection-Reconstruction-Pipeline
```

### Create Environment

```bash
python -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Detection Training

Train the DeBERTa-based detection model:

```bash
python train_deberta_detection.py
```

Output:

```text
best_model/
metrics_summary.txt
val_predictions.csv
test_predictions.csv
test_predictions_thresholded.csv
```

---

## Reconstruction Training

Train the BART reconstruction model:

```bash
python train_bart_reconstruction.py
```

Output:

```text
generated_predictions.csv
rouge_scores.txt
bertscore_results.txt
```

---

## Example

### Input Argument

```text
Renewable energy should replace fossil fuels.
[MASK]
Therefore, governments should increase investments in solar energy.
```

### Generated Output

```text
Fossil fuels contribute significantly to environmental pollution and climate change.
```

---

## Experimental Setup

| Parameter            | Value        |
| -------------------- | ------------ |
| Detection Model      | DeBERTa Base |
| Reconstruction Model | BART Large   |
| Learning Rate        | 1e-5         |
| Batch Size           | 16           |
| Epochs               | 24           |
| Maximum Length       | 256          |
| Weight Decay         | 0.01         |
| Warmup Ratio         | 0.10         |
| Early Stopping       | 3            |

---

## Research Contributions

* End-to-end enthymeme processing pipeline
* Transformer-based detection framework
* Threshold optimization strategy
* Mask-based reconstruction methodology
* Automated evaluation framework
* Reproducible NLP experimentation environment

---

## Future Work

* Larger Language Models (LLMs)
* T5-based reconstruction
* LLaMA-based reconstruction
* Retrieval-Augmented Generation (RAG)
* Multi-gap enthymeme reconstruction
* Cross-domain argument mining
* Explainable reconstruction models

---

## Author

**Nagaraju Saggu**

Master's Student

Bauhaus University Weimar

Research Areas:

* Natural Language Processing (NLP)
* Large Language Models (LLMs)
* Argument Mining
* Machine Learning
* Deep Learning
* Structural Health Monitoring

---

## Acknowledgements

This project was inspired by recent research on automated enthymeme detection and reconstruction in argumentative texts and learner essays.

---

## License

This project is released under the MIT License.

Feel free to use, modify, and distribute this work for research and educational purposes.
