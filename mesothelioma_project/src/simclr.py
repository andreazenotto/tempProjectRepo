import tensorflow as tf
import os
import numpy as np
from PIL import Image


def load_image(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = img / 255.0
    return img


def augment(image):
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random crop
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    image = tf.image.random_crop(image, size=[224, 224, 3])
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(minval=0, maxval=4, dtype=tf.int32))
    # Random brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


def process_image(image_path):
    image = load_image(image_path)
    augmented_image1 = augment(image)
    augmented_image2 = augment(image)
    return augmented_image1, augmented_image2


def create_dataset(directory, batch_size=32):
    all_images = []

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            # For each subfolder (WSI) inside the class folder
            for wsi_dir in os.listdir(class_path):
                wsi_path = os.path.join(class_path, wsi_dir)
                if os.path.isdir(wsi_path):
                    # Add all images from that WSI
                    for img_name in os.listdir(wsi_path):
                        img_path = os.path.join(wsi_path, img_name)
                        all_images.append(img_path)

    # Create a dataset from the list of images
    path_ds = tf.data.Dataset.from_tensor_slices(all_images)
    # Apply the preprocessing function and create pairs of images
    image_ds = path_ds.map(lambda x: process_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    # Add batching and shuffle
    image_ds = image_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_ds


def build_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = base_model(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(features)
    outputs = tf.keras.layers.Dense(128)(x)
    full_model = tf.keras.Model(inputs, outputs)
    return full_model, base_model


def nt_xent_loss(proj_1, proj_2, temperature=0.5):
    batch_size = tf.shape(proj_1)[0]

    # Normalize
    proj_1 = tf.math.l2_normalize(proj_1, axis=1)
    proj_2 = tf.math.l2_normalize(proj_2, axis=1)

    projections = tf.concat([proj_1, proj_2], axis=0)

    logits = tf.matmul(projections, projections, transpose_b=True) / temperature

    # Mask self-similarity
    mask = tf.eye(2 * batch_size)
    logits = logits * (1. - mask) - 1e9 * mask

    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def train_simclr(dataset, epochs):
    model, base_model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for step, (view1, view2) in enumerate(dataset):
            with tf.GradientTape() as tape:
                proj1 = model(view1, training=True)
                proj2 = model(view2, training=True)
                loss = nt_xent_loss(proj1, proj2)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss.numpy()

        avg_loss = total_loss / (step + 1)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    base_model.save_weights("resnet50_simclr_weights.h5")
    