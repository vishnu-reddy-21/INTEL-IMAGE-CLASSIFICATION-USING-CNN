#!/usr/bin/env python3
"""
Train a CNN on Intel Image Classification dataset using transfer learning (ResNet50).

Example:
  python train.py --data-dir data/Intel/seg_train --val-dir data/Intel/seg_test --epochs 8 --batch-size 32 --out outputs

Saves:
 - outputs/model.h5
 - outputs/class_indices.json
"""
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def create_model(img_size, num_classes):
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(img_size, img_size, 3))
    base.trainable = False   # Phase 1 freeze

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model, base

def data_loaders(data_dir, val_dir, img_size, batch_size):

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.6, 1.4],
        zoom_range=0.3,
        shear_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = valid_aug.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen


def train(args):

    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs

    train_gen, val_gen = data_loaders(args.data_dir, args.val_dir,
                                      img_size, batch_size)

    num_classes = len(train_gen.class_indices)

    # Save class mapping
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)

    # Compute class weights
    labels = train_gen.classes
    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(labels),
                              y=labels)
    class_weights = dict(enumerate(cw))

    model, base = create_model(img_size, num_classes)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.3,
      patience=3,
      min_lr=1e-6,
      verbose=1
)

    # -------- PHASE 1 ----------
    print("\n🔵 PHASE 1: Training top layers only...\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    epochs_phase1 = max(5, epochs // 2)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=[reduce_lr]
    )

    # -------- PHASE 2 ----------
    print("\n🟢 PHASE 2: Fine-tuning ResNet50 (last 40 layers)...\n")

    base.trainable = True

    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
         validation_data=val_gen,
         epochs=epochs,
         class_weight=class_weights,
         callbacks=[reduce_lr] 
    )

    model.save(os.path.join(args.out, "model.h5"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()
    train(args)


