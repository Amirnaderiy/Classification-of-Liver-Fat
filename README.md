# Liver fat grading from B-mode ultrasound with transfer learning

Deep-learning baselines for classifying non-alcoholic fatty liver disease (NAFLD) into four fat grades from B-mode ultrasound images.

| Grade | Hepatic fat |
|---|---|
| 0 | Healthy (< 5%) |
| 1 | Mild (5–30%) |
| 2 | Moderate (30–70%) |
| 3 | Severe (> 70%) |

This repository contains two scripts:

| Script | What it does |
|---|---|
| `train_efficientnet.py` | End-to-end classifier. An ImageNet EfficientNet-B7 backbone is first frozen while a new head is trained, then its top layers are fine-tuned. Reports test accuracy, a per-class report and a confusion matrix. |
| `extract_cnn_features.py` | Uses a pretrained InceptionResNetV2 as a fixed feature extractor and saves variance-filtered deep features for use with classical classifiers (random forest, XGBoost, ...). |

The full hybrid pipeline (seven pretrained CNNs + GLCM radiomics + feature selection + gradient boosting) is in [Classification-of-the-grade-of-Non-alcoholic-fatty-liver](https://github.com/Amirnaderiy/Classification-of-the-grade-of-Non-alcoholic-fatty-liver).

## Installation

```bash
git clone https://github.com/Amirnaderiy/Classification-of-Liver-Fat.git
cd Classification-of-Liver-Fat
pip install -r requirements.txt
```

## Data

The ultrasound dataset is not included. Arrange your images in one folder per class:

```
data/
  train/  grade0/  grade1/  grade2/  grade3/
  val/    grade0/  grade1/  grade2/  grade3/
  test/   grade0/  grade1/  grade2/  grade3/
```

Split by patient, not by image, so that images from the same person never appear in both training and test sets.

## Usage

Train and evaluate the EfficientNet-B7 classifier:

```bash
python train_efficientnet.py --train-dir data/train --val-dir data/val --test-dir data/test --out-dir outputs
```

Outputs: `training_curves.png`, `test_report.txt` and the trained model (`.keras`).

Extract deep features for a classical classifier:

```bash
python extract_cnn_features.py --train-dir data/train --test-dir data/test --out features.npz
```

Run either script with `--help` to see all options (epochs, batch size, number of layers to fine-tune, variance threshold).

## Related publications

- Naderi Yaghouti AR, Shalbaf A. Classification of nonalcoholic fatty liver grades using pre-trained convolutional neural networks and a random forest classifier on B-mode ultrasound images. *Journal of Biomedical Physics and Engineering*, 2025. [Link](https://jbpe.sums.ac.ir/article_49907.html)
- Naderi Yaghouti AR, Shalbaf A, Maghsoudi A. Automatic classification of non-alcoholic fatty liver using texture features from ultrasound images. *Tehran University Medical Journal* 79(1):10–17, 2021. [Link](https://tumj.tums.ac.ir/article-1-11130-en.html)

## Contact

Amir Reza Naderi Yaghouti · [amirnaderi.com](https://amirnaderi.com) · naderi.bme@gmail.com
