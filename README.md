# Breast Ultrasound Classification: CNNs and Transfer Learning

**Comparing a custom CNN with ImageNet-pretrained models on the BUSI dataset.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidecaramia/breast-ultrasound-deep-learning/blob/main/notebooks/breast_ultrasound_classification.ipynb)

## Overview

This project explores the classification of breast ultrasound images into three categories: **benign**, **malignant** and **normal**.

Using the **Breast Ultrasound Images (BUSI)** dataset, the study compares a convolutional neural network trained from scratch with four pretrained architectures used for transfer learning.

The experiments examine the challenges of learning from a small, imbalanced dataset, with particular attention to class-specific performance and errors involving malignant images.

> This is an academic deep learning project. It is not a medical device and must not be used for diagnosis or clinical decision-making.

The complete project presentation is available in [`report/project_presentation.pdf`](report/project_presentation.pdf).

## Dataset and preprocessing

BUSI contains **780 ultrasound images** across the three target classes.

| Property | Configuration |
|---|---|
| Task | Three-class image classification |
| Classes | Benign, malignant, normal |
| Dataset size | 780 images |
| Model input | 224 × 224 RGB images |
| Training partition | Approximately 70% |
| Validation partition | Approximately 15% |
| Test partition | Approximately 15% |
| Split strategy | Performed separately within each class |
| Segmentation masks | Excluded from the classification inputs |

The dataset is distributed separately from this repository.

BUSI was introduced by Al-Dhabyani et al. and contains images collected from **600 women aged 25–75**.

**Reference:**  
W. Al-Dhabyani, M. Gomaa, H. Khaled and A. Fahmy, “Dataset of breast ultrasound images,” *Data in Brief*, 2020.  
[https://doi.org/10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)

## Experimental approach

The experiments follow two complementary approaches:

1. Build and evaluate a compact CNN trained from scratch.
2. Compare it with pretrained networks used as frozen feature extractors.

Class imbalance is addressed through class weights, alongside additional experiments involving oversampling and data augmentation.

### Custom convolutional network

The custom architecture consists of:

- Three convolutional blocks.
- Global average pooling.
- A three-unit output layer with softmax activation.

Five CNN variants explore different combinations of augmentation, dropout and class-balancing strategies.

### Transfer learning

Four ImageNet-pretrained architectures are evaluated:

| Backbone | Training approach |
|---|---|
| VGG16 | Frozen backbone with a task-specific classification head |
| EfficientNetB0 | Frozen backbone with a task-specific classification head |
| DenseNet121 | Frozen backbone with a task-specific classification head |
| MobileNetV3Small | Frozen backbone with a task-specific classification head |

The pretrained backbones are not fine-tuned end-to-end in these experiments.

## Evaluation

Models are assessed using:

- Accuracy.
- Class-wise precision, recall and F1 score.
- Macro-averaged F1 score.
- Confusion matrices.
- One-vs-rest ROC curves.

The final comparison was conducted once on a held-out test set containing **118 images**.

## Results

**MobileNetV3Small** achieved the strongest overall test performance in the reported comparison.

| Model | Test accuracy | Macro F1 | Malignant recall | Malignant F1 |
|---|---:|---:|---:|---:|
| CNN trained from scratch | 0.67 | 0.56 | 0.47 | 0.58 |
| **MobileNetV3Small** | **0.82** | **0.81** | **0.66** | **0.71** |

Compared with the custom CNN, MobileNetV3Small improved both overall performance and the reported malignant-class metrics.

However, some malignant images were still classified as benign. This illustrates why overall accuracy must be considered alongside class-specific errors.

### Confusion matrices

| CNN trained from scratch | MobileNetV3Small |
|---|---|
| ![Custom CNN confusion matrix](assets/cnn_scratch_confusion_matrix.png) | ![MobileNetV3Small confusion matrix](assets/mobilenet_confusion_matrix.png) |

### ROC curves

| CNN trained from scratch | MobileNetV3Small |
|---|---|
| ![Custom CNN one-vs-rest ROC curves](assets/cnn_scratch_roc.png) | ![MobileNetV3Small one-vs-rest ROC curves](assets/mobilenet_roc.png) |

## Tools and technologies

| Tool | Purpose |
|---|---|
| **Python** | Experiment implementation and data processing |
| **TensorFlow / Keras** | CNN development, pretrained backbones and model training |
| **NumPy** | Numerical operations |
| **scikit-learn** | Evaluation metrics |
| **Matplotlib** | Visualization |
| **Pillow** | Image handling |
| **Google Colab and Google Drive** | Notebook execution and storage of data and training outputs |

Python dependencies are listed in [`requirements.txt`](requirements.txt).

## Repository contents

| Path | Description |
|---|---|
| `notebooks/breast_ultrasound_classification.ipynb` | Complete experimental notebook |
| `assets/cnn_scratch_confusion_matrix.png` | Confusion matrix for the custom CNN |
| `assets/cnn_scratch_roc.png` | ROC curves for the custom CNN |
| `assets/mobilenet_confusion_matrix.png` | Confusion matrix for MobileNetV3Small |
| `assets/mobilenet_roc.png` | ROC curves for MobileNetV3Small |
| `report/project_presentation.pdf` | Project presentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |
| `README.md` | Project overview and execution instructions |

The dataset, augmented images, trained model checkpoints and Google Drive files are not included in this repository.

## Running the notebook

The original workflow was developed in **Google Colab** and uses **Google Drive** to store datasets, checkpoints and training histories.

1. Obtain the BUSI dataset through the dataset reference above.
2. Create a folder named `DEEP_LEARNING_PROJECT` in Google Drive.
3. Extract the dataset into the following location:

   ```text
   DEEP_LEARNING_PROJECT/Dataset_BUSI_with_GT/
   ```

4. Open [`notebooks/breast_ultrasound_classification.ipynb`](notebooks/breast_ultrasound_classification.ipynb) in Google Colab.
5. Connect Google Drive when requested.
6. Execute the notebook cells in order.

The notebook creates the training, validation and test directories, together with the required model directories.

Running all experiments from scratch can take considerable time. If checkpoints already exist in the expected Google Drive folders, the notebook loads them automatically.

## Limitations

- **Dataset size and imbalance:** limited training data makes learning robust features from scratch challenging.
- **External validity:** evaluation uses a single dataset; performance on external datasets has not been established.
- **Frozen pretrained features:** transfer-learning backbones are used as feature extractors rather than being fully fine-tuned.
- **Malignant-class errors:** some malignant images are predicted as benign, limiting the practical interpretation of the reported accuracy.

## Authors

Developed collaboratively by:

- [Davide Francesco Caramia](https://github.com/davidecaramia)
- [Nicolò Bachiorri](https://github.com/nicolobachiorri)
- [Hervé Mottaran](https://github.com/hervemottaran)

All team members contributed jointly to data preparation, model development, experimental evaluation, interpretation and presentation.

**Foundations of Deep Learning — Università degli Studi di Milano-Bicocca**  
**Academic Year 2025/2026**
