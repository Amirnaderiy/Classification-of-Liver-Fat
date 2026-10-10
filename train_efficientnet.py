"""Four-grade liver fat classification from B-mode ultrasound with EfficientNet-B7.

Two-stage transfer learning:
  1. Train a new classification head on top of a frozen ImageNet EfficientNet-B7.
  2. Unfreeze the top layers of the backbone and fine-tune with a small learning rate.

The data folders must follow the Keras ``flow_from_directory`` layout, one
sub-folder per class, for example::

    data/train/grade0/*.png   data/train/grade1/*.png   ...
    data/val/grade0/*.png     ...
    data/test/grade0/*.png    ...

Example::

    python train_efficientnet.py --train-dir data/train --val-dir data/val --test-dir data/test
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=250, help="epochs for stage 1 (frozen backbone)")
    p.add_argument("--finetune-epochs", type=int, default=100, help="epochs for stage 2 (fine-tuning)")
    p.add_argument("--unfreeze-last", type=int, default=30, help="number of backbone layers to unfreeze in stage 2")
    p.add_argument("--weights", default="imagenet", help="backbone weights: 'imagenet' or 'none'")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_generators(args):
    # EfficientNet in Keras expects raw 0-255 pixels; preprocess_input handles scaling.
    # The same preprocessing is used for training, validation and test images.
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    common = dict(target_size=IMG_SIZE, class_mode="categorical", seed=args.seed)
    train = train_gen.flow_from_directory(args.train_dir, batch_size=args.batch_size, shuffle=True, **common)
    val = eval_gen.flow_from_directory(args.val_dir, batch_size=16, shuffle=False, **common)
    test = eval_gen.flow_from_directory(args.test_dir, batch_size=16, shuffle=False, **common)
    return train, val, test


def build_model(num_classes, weights):
    backbone = EfficientNetB7(
        weights=None if weights.lower() == "none" else weights,
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )
    backbone.trainable = False

    model = models.Sequential(
        [
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, backbone


def step_decay(epoch, initial_lr=1e-3, drop=0.5, epochs_drop=50):
    return initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


def unfreeze_top(backbone, n_layers):
    """Unfreeze the last ``n_layers`` of the backbone; BatchNorm layers stay frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def plot_history(histories, out_dir):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    split = len(histories[0].history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tr, va, name) in zip(axes, [(acc, val_acc, "Accuracy"), (loss, val_loss, "Loss")]):
        ax.plot(epochs, tr, label=f"Training {name.lower()}")
        ax.plot(epochs, va, label=f"Validation {name.lower()}")
        ax.axvline(split, color="grey", linestyle="--", linewidth=1, label="Start of fine-tuning")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = make_generators(args)
    num_classes = train.num_classes
    model, backbone = build_model(num_classes, args.weights)

    # Stage 1: train the head only.
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(
        train,
        epochs=args.head_epochs,
        validation_data=val,
        callbacks=[LearningRateScheduler(step_decay)],
    )

    # Stage 2: fine-tune the top of the backbone.
    unfreeze_top(backbone, args.unfreeze_last)
    model.compile(optimizer=RMSprop(learning_rate=5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train, epochs=args.finetune_epochs, validation_data=val)

    plot_history([h1, h2], out_dir)
    model.save(out_dir / "efficientnetb7_liver_fat.keras")

    # Test-set evaluation.
    probs = model.predict(test)
    y_pred = np.argmax(probs, axis=1)
    y_true = test.classes
    class_names = [k for k, _ in sorted(test.class_indices.items(), key=lambda kv: kv[1])]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print("Confusion matrix:\n", cm)
    (out_dir / "test_report.txt").write_text(report + "\nConfusion matrix:\n" + np.array2string(cm) + "\n")


if __name__ == "__main__":
    main()
