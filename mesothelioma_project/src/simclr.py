import tensorflow as tf
import os


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random crop + resize
    image = tf.image.random_crop(image, size=[int(0.8 * image.shape[0]), int(0.8 * image.shape[1]), 3])
    image = tf.image.resize(image, [image.shape[0], image.shape[1]])

    # Color jitter
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_hue(image, max_delta=0.05)

    return image


def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    view1 = augment(image)
    view2 = augment(image)
    return view1, view2


def shuffle_and_batch(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_dataset(directory):
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


def nt_xent_loss(proj_1, proj_2, temperature=0.1):
    batch_size = tf.shape(proj_1)[0]

    # L2 normalization for projections (cosine similarity)
    proj_1 = tf.math.l2_normalize(proj_1, axis=1)
    proj_2 = tf.math.l2_normalize(proj_2, axis=1)

    # Concatenate projections: [p1_1, ..., p1_B, p2_1, ..., p2_B] -> (2B, D)
    projections = tf.concat([proj_1, proj_2], axis=0)

    # Compute logits: similarity between all pairs (sim(z_i, z_j)/τ)
    similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
    logits = similarity_matrix / temperature  # (2B, 2B)

    # Mask self-similarity
    mask = tf.eye(2 * batch_size)
    logits = logits * (1. - mask) - 1e9 * mask

    # Create labels: the positive for i is i + B if i < B, or i - B
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def train_simclr(dataset, epochs=100, batch_size=512):
    model, base_model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(epochs):
        # Shuffle and batch the dataset
        dataset = shuffle_and_batch(dataset, batch_size)

        # Initialize the total loss for this epoch
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
    