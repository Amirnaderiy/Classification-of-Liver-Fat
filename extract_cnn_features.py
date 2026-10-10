"""Extract deep features from ultrasound images with a pretrained InceptionResNetV2.

The convolutional base (ImageNet weights, no top) turns each image into a
18 x 12 x 1536 feature map, which is flattened and filtered with a variance
threshold. The resulting feature matrices can be used with any classical
classifier (random forest, XGBoost, SVM, ...).

The variance threshold is fitted on the training features only and then applied
to the test features, so both sets keep the same columns.

Example::

    python extract_cnn_features.py --train-dir data/train --test-dir data/test --out features.npz
"""

import argparse

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (636, 444)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--out", default="features.npz")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--variance-threshold", type=float, default=0.8 * (1 - 0.8))
    p.add_argument("--weights", default="imagenet", help="backbone weights: 'imagenet' or 'none'")
    return p.parse_args()


def extract_features(conv_base, directory, batch_size):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
        class_mode="sparse",
    )
    features = conv_base.predict(generator)
    features = features.reshape(len(features), -1)
    return features, generator.classes, generator.class_indices


def main():
    args = parse_args()
    conv_base = InceptionResNetV2(
        weights=None if args.weights.lower() == "none" else args.weights,
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )

    X_train, y_train, classes = extract_features(conv_base, args.train_dir, args.batch_size)
    X_test, y_test, _ = extract_features(conv_base, args.test_dir, args.batch_size)

    selector = VarianceThreshold(threshold=args.variance_threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    print(f"Features per image: {X_train.shape[1]} -> {X_train_sel.shape[1]} after variance threshold")

    np.savez_compressed(
        args.out,
        X_train=X_train_sel,
        y_train=y_train,
        X_test=X_test_sel,
        y_test=y_test,
        class_names=np.array(sorted(classes, key=classes.get)),
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
