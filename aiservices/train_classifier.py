"""Simple multi-class image classifier training script for beerbotplus.

This script trains a custom classifier using transfer learning (MobileNetV2).
It's designed to be beginner-friendly and expandable to 2-4 classes.

Usage:
    python train_classifier.py --data-dir ./data/train --epochs 15 --output beer_classifier.h5

Data folder structure expected:
    data/
      train/
        beer/              # images of people drinking beer
        non_beer_drink/    # images of people drinking other drinks
        water/             # images of people drinking water (optional)
        unrelated/         # images unrelated to drinking (optional)

Each subfolder should contain ~50-200 images (JPG/PNG).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(num_classes: int, input_size: Tuple[int, int] = (224, 224)):
    """Build a transfer-learning model using MobileNetV2.

    Args:
        num_classes: Number of output classes (2, 3, 4, etc.)
        input_size: Input image size (H, W)

    Returns:
        Compiled Keras model
    """
    # Load pretrained MobileNetV2 (ImageNet weights, no top layers)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*input_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model to use as feature extractor
    base_model.trainable = False

    # Build simple classifier on top
    if num_classes == 2:
        # Binary classification: sigmoid, binary_crossentropy
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        output_units = 1
    else:
        # Multi-class: softmax, categorical_crossentropy
        loss = "categorical_crossentropy"
        output_activation = "softmax"
        output_units = num_classes

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(output_units, activation=output_activation)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"]
    )

    return model

def train_classifier(
    data_dir: str,
    output_path: str,
    epochs: int = 15,
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    validation_split: float = 0.2
) -> None:
    """Train a multi-class image classifier.

    Args:
        data_dir: Path to data/train directory with class subdirectories
        output_path: Where to save the trained model (.h5 file)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size (H, W)
        validation_split: Fraction of data to use for validation (0.0-1.0)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Count classes from subdirectories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    num_classes = len(class_dirs)
    logger.info(f"Found {num_classes} classes: {[d.name for d in class_dirs]}")

    if num_classes < 2:
        raise ValueError("Need at least 2 classes. Create subdirectories in your data_dir.")

    # Data augmentation (for training robustness)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )

    # Load training data
    logger.info("Loading training data...")
    train_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical" if num_classes > 2 else "binary",
        subset="training"
    )

    # Load validation data
    val_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical" if num_classes > 2 else "binary",
        subset="validation"
    )

    # Build and train model
    logger.info(f"Building model for {num_classes} classes...")
    model = build_model(num_classes=num_classes, input_size=img_size)

    logger.info(f"Training for {epochs} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=train_data.samples // batch_size
    )

    # Save model
    logger.info(f"Saving model to {output_path}")
    model.save(output_path)

    # Print summary
    logger.info("Training complete!")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.2%}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a multi-class image classifier using transfer learning"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/train",
        help="Path to training data directory with class subdirectories"
    )
    parser.add_argument(
        "--output",
        default="beer_classifier.h5",
        help="Path where to save the trained model (.h5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (height width)"
    )

    args = parser.parse_args()
    train_classifier(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
