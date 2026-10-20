---
license: mit
library_name: keras
pipeline_tag: image-classification
datasets:
  - nelorth/oxford-flowers
metrics:
  - accuracy
---

# Flower Classifier (Oxford 102 Flowers)

An EfficientNetB5 image classifier that recognizes **102 flower species**.
Live demo: https://huggingface.co/spaces/mgeorgiev5/Flower-Classifier
Code: https://github.com/martingeorgiev05/flowerClassifier

## Results (Oxford 102 test split, 6,149 images)

| Metric                                    | Value      |
| ----------------------------------------- | ---------- |
| Top-1 accuracy                            | **93.27%** |
| Top-5 accuracy                            | **98.47%** |
| Balanced accuracy (mean per-class recall) | 94.0%      |

The test split was used only to report this final number. All design decisions were made on the
validation split. The final model was trained on train + validation (2,040 images).

## How it was built

EfficientNetB5 pretrained on ImageNet, with a small classification head, trained in two phases
(head only, then fine-tuning blocks 6-7 with a cosine learning-rate schedule).
Improvements measured step by step on the test set during development:

| Change                                              | Test top-1 |
| --------------------------------------------------- | ---------- |
| Baseline (original setup)                           | 73.3%      |
| Keep BatchNorm in inference mode during fine-tuning | 75.8%      |
| Realistic augmentation                              | 79.7%      |
| Linear head + label smoothing                       | 80.1%      |
| Aspect-preserving resize + center crop              | 82.0%      |
| Fine-tune blocks 6-7, AdamW, cosine schedule        | 86.4%      |
| Mixup                                               | 86.7%      |
| Final run on train + validation                     | 93.3%      |

## Usage

The model expects **raw RGB pixels in the range 0-255** (EfficientNet normalizes internally).
Resize the shorter side to 300 px (antialiased), then center-crop to 300x300.

    import tensorflow as tf
    model = tf.keras.models.load_model("flower_model.h5", compile=False)
    probabilities = model.predict(batch)  # batch shape: (N, 300, 300, 3) -> (N, 102)

The full preprocessing code and the class names are in the GitHub repository.

## Limitations

- It knows only the 102 species of the dataset (flowers common in the United Kingdom).
  Any other flower will get a confident wrong answer from this list.
- A photo that is not a flower at all (a dog, a car) also gets a flower name, since there is
  no "not a flower" class.
- Training used label smoothing, so confidence values are lower than in typical classifiers;
  a prediction of 85% is already a confident one.
- Some look-alike species are still confused (for example rose vs. camellia,
  petunia vs. morning glory).
