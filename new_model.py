import os
import io
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import itertools

os.environ["TF_KERAS"] = "1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU acceleration enabled with TensorFlow-Metal.")
    except RuntimeError as e:
        print(e)

os.makedirs('quickdraw_images', exist_ok=True)


def download_and_save_category(category, base_dir, num_samples=1000):
    """
    Презема слики од Quick, Draw! и ги зачувува како PNG слики.
    """
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
    print(f"Се симнуваат податоците за категоријата: '{category}'")

    response = requests.get(url)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))

    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    for i, sample in enumerate(data[:num_samples]):
        img_array = sample.reshape(IMG_SIZE).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        file_path = os.path.join(category_dir, f"{category}_{i}.png")
        img.save(file_path)
    print(f"Зачувани се {num_samples} слики за: '{category}'")


def prepare_dataset(categories, base_dir, num_samples):
    """
    За дополнување на датасетот за тренинг
    """
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir) or len(os.listdir(category_dir)) < num_samples:
            download_and_save_category(category, base_dir, num_samples)
        else:
            print(f"Податоците за '{category}' веќе постојат.")


BASE_DATA_DIR = "quickdraw_images"

CATEGORIES = [
    "airplane", "apple", "banana", "bed", "bicycle", "bird", "cake", "car", "cat", "chair",
    "clock", "dog", "fish", "flower", "frog", "guitar", "hat", "house", "tree", "sun"
]
NUM_SAMPLES = 5000  # Број на слики за тренирање за секој категорија
IMG_SIZE = (28, 28)  # Сликите од Quick, Draw! се 28x28 пиксели
BATCH_SIZE = 64
EPOCHS = 8
VALIDATION_SPLIT = 0.2
SEED = 123

os.makedirs(BASE_DATA_DIR, exist_ok=True)


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def plot_metrics(history):
    """ Plot training & validation accuracy and loss curves """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """ Plot a confusion matrix """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\n📂 Loading image dataset...")
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1. / 255,
        validation_split=VALIDATION_SPLIT
    )

    train_dataset = data_gen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode="grayscale",
        subset="training"
    )

    validation_dataset = data_gen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode="grayscale",
        subset="validation"
    )

    num_classes = len(CATEGORIES)
    input_shape = IMG_SIZE + (1,) # (height, width, channels)
    model = build_model(input_shape, num_classes)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    loss, accuracy = model.evaluate(validation_dataset)
    print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

    full_model_path = "full_doodle_model.h5"
    model.save(full_model_path, save_format="h5")
    print(f"✅ Model saved at: {full_model_path}")

    # Plot training metrics
    plot_metrics(history)

    # Generate Confusion Matrix
    print("\n📊 Generating Confusion Matrix...")
    y_true = []
    y_pred = []
    for images, labels in validation_dataset:
        preds = np.argmax(model.predict(images), axis=1)
        y_true.extend(labels)
        y_pred.extend(preds)
        if len(y_true) >= validation_dataset.samples:
            break

    plot_confusion_matrix(y_true, y_pred, CATEGORIES)


if __name__ == '__main__':
    main()
