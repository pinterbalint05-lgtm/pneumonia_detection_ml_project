# Pneumonia Detection from Chest X-rays using a VGG16-style CNN

This project uses a convolutional neural network to classify chest X-ray images as either **Normal** or **Pneumonia**. The model is based on the VGG16 architecture and is trained from scratch for binary image classification.

The goal of the project is not only to train a working CNN, but also to handle the dataset correctly, evaluate the model with suitable metrics, and understand the tradeoff between detecting pneumonia cases and avoiding false alarms.

## Project overview

- **Task:** binary classification of chest X-ray images
- **Classes:**
  - `0` = Normal
  - `1` = Pneumonia
- **Model:** VGG16-style CNN trained from scratch
- **Input image size:** 224 × 224 RGB image
- **Main notebook:** `main.ipynb`
- **Framework:** TensorFlow / Keras

The final layer of the model uses a single sigmoid neuron, so the output is interpreted as the predicted probability of pneumonia.

## Dataset

The images and labels come from the **RSNA Pneumonia Detection Challenge** dataset. In this notebook, the data is not downloaded directly from the RSNA website; it is accessed through the following Kaggle dataset package:

```text
stevezeyuzhang/rsna-pneumonia-detection-challenge
```

So, the original challenge is RSNA, but the actual dataset path used by the notebook is the Kaggle package above. This is also why the notebook paths contain:

```text
/kaggle/input/datasets/stevezeyuzhang/rsna-pneumonia-detection-challenge/...
```

The dataset is not included in this repository because it is too large. The notebook expects the Kaggle dataset folder to contain files such as:

```text
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
stage_2_train_images_png/
stage_2_test_images_png/
```

## Data handling

The labelled file `stage_2_train_labels.csv` is used for measurable training and evaluation. The external Kaggle test image folder contains images, but the provided files do not contain the true labels for those external test images.

Because of this, the labelled data is split into:

- **training set**: used to train the neural network,
- **validation set**: used for model selection and threshold tuning,
- **internal test set**: used only for final measurable evaluation.

The external test images are used only to generate predictions, not to calculate accuracy, recall, F1-score, or a confusion matrix.

Final data split used in the notebook:

```text
Total labelled patients : 26684
Normal patients         : 20672
Pneumonia patients      : 6012
Training images         : 18678
Validation images       : 4003
Internal test images    : 4003
External test images    : 3000
```

## Model architecture

The model follows a VGG16-style CNN structure:

- convolutional blocks with 3 × 3 filters,
- max pooling after convolutional blocks,
- batch normalization after pooling layers,
- fully connected dense layers at the end,
- dropout to reduce overfitting,
- sigmoid output for binary classification.

The model is trained from scratch. Pretrained ImageNet weights are not used in the final notebook.

## Training setup

Important training settings:

```text
Image size      : 224 × 224
Batch size      : 64
Epochs          : 50
Learning rate   : 0.00001
Optimizer       : Adam
Loss function   : binary cross-entropy
Checkpoint      : best validation PR-AUC
Early stopping  : validation PR-AUC, patience = 10
```

Class weights are used because the dataset is imbalanced. There are many more normal cases than pneumonia cases, so pneumonia samples are weighted more during training.

The model tracks the following metrics:

- accuracy,
- precision,
- recall,
- ROC-AUC,
- PR-AUC.

PR-AUC is especially useful here because pneumonia is the minority class.

## Threshold selection

The model outputs pneumonia probabilities. To convert probabilities into final class labels, a threshold is needed:

```text
probability >= threshold  -> Pneumonia
probability < threshold   -> Normal
```

Instead of only using the default threshold of `0.50`, the notebook tests several thresholds on the validation set and selects the threshold with the best F1-score.

The selected threshold in the final run was:

```text
Best validation threshold by F1-score: 0.65
```

A lower fixed threshold of `0.45` is also tested to demonstrate the medical tradeoff. Lowering the threshold increases pneumonia recall, but also increases the number of false positives.

## Final internal test results

The main final result uses the validation-selected threshold of `0.65`.

```text
Internal test results, threshold = 0.65

Normal:
  precision: 0.90
  recall:    0.83
  f1-score:  0.86

Pneumonia:
  precision: 0.54
  recall:    0.69
  f1-score:  0.60

Overall accuracy: 0.80
Macro average F1-score: 0.73
Weighted average F1-score: 0.80
```

Confusion matrix for threshold `0.65`:

```text
[[2563  538]
 [ 281  621]]
```

Interpretation:

- 2563 normal X-rays were correctly classified as normal,
- 538 normal X-rays were incorrectly classified as pneumonia,
- 281 pneumonia X-rays were incorrectly classified as normal,
- 621 pneumonia X-rays were correctly classified as pneumonia.

The notebook also evaluates a recall-focused threshold of `0.45`.

```text
Internal test results, threshold = 0.45

Pneumonia recall: 0.87
Pneumonia F1-score: 0.56
Overall accuracy: 0.69
```

This threshold detects more pneumonia cases, but it produces more false positives. The main reported threshold remains `0.65` because it gives the better F1-score balance.

## Generated outputs

The notebook creates the following useful outputs:

```text
models/vgg16_pneumonia_best.keras
vgg16_pneumonia_best.keras
training_curves_updated.png
external_test_predictions.csv
```

The model file is not included in the GitHub repository because it is large. It should be downloaded separately from the Kaggle notebook output if needed.

## How to run the project

The easiest way to run the project is on Kaggle with GPU enabled.

1. Open `main.ipynb` on Kaggle.
2. Make sure the RSNA pneumonia dataset is attached.
3. Enable GPU acceleration.
4. Run the notebook cells in order.
5. Check the final internal test metrics and generated prediction CSV.

If running locally, update the dataset paths in the configuration cell:

```python
BASE_DIR = "path/to/rsna-pneumonia-detection-challenge"
```

Then make sure the required files and image folders are available under that path.

## Requirements

Main Python libraries used:

```text
numpy
pandas
matplotlib
tensorflow / keras
scikit-learn
kagglehub
```

## Notes and limitations

- The external Kaggle test images do not include ground-truth labels in the provided files, so final measurable evaluation is done on the internal labelled test split.
- The dataset is imbalanced, so accuracy alone is not enough to judge model performance.
- Recall is important because a false negative means a pneumonia case was missed.
- Precision is also important because low precision means many normal X-rays are flagged as pneumonia.
- The model is useful as a machine learning project demonstration, but it is not suitable for real medical diagnosis without much more validation.

## Repository contents

The repository contains the main notebook and project documentation:

```text
README.md
main.ipynb
.gitignore
```
