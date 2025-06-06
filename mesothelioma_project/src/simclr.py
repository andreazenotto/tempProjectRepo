import os
import tensorflow as tf
import keras_hub


def add_gaussian_noise(image, mean=0.0, stddev=10.0):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
    image = tf.cast(image, tf.float32)
    noisy_image = image + noise
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 255.0)
    return tf.cast(noisy_image, tf.uint8)


def augment(image):
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random crop
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, size=[224, 224, 3])
    # Random brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # gaussian noise
    image = add_gaussian_noise(image)

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
 

def build_model(version='resnet_18_imagenet'): 
    # version = 'resnet_18_imagenet' or 'resnet_50_imagenet'
    base_model = keras_hub.models.ResNetBackbone.from_preset(
        version,
        input_shape=(224, 224, 3),
        include_rescaling=False
    )
    
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = base_model(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)

    # Projection head come nel paper
    if version == 'resnet_18_imagenet':
        x = tf.keras.layers.Dense(512, activation='relu')(features)
    elif version == 'resnet_50_imagenet':
        x = tf.keras.layers.Dense(2048, activation='relu')(features)
    outputs = tf.keras.layers.Dense(128)(x)

    full_model = tf.keras.Model(inputs, outputs)
    return full_model, base_model


class SimCLRTrainer(tf.keras.Model):
    def __init__(self, encoder, temperature):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        view1, view2 = data  # unpack the two views

        with tf.GradientTape() as tape:
            proj1 = self.encoder(view1, training=True)
            proj2 = self.encoder(view2, training=True)
            loss = nt_xent_loss(proj1, proj2, self.temperature)

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        return {"loss": loss}

def nt_xent_loss(proj_1, proj_2, temperature):
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


def train_simclr(dataset, resnet_version='resnet_18_imagenet', epochs=50, batch_size=256, temperature=0.1, lr=2e-4, lr_decay=True):
    strategy = tf.distribute.MirroredStrategy()
    dataset = shuffle_and_batch(dataset, batch_size)

    def lr_scheduler(epoch):
        factor = pow((1 - (epoch / epochs)), 0.9)
        return lr * factor

    with strategy.scope():
        full_model, base_model = build_model(resnet_version)
        simclr_model = SimCLRTrainer(full_model, temperature)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
        simclr_model.compile(optimizer=optimizer)

        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # Callbacks
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_simclr_model.h5',
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=True
        )

        callbacks = [checkpoint_callback]
        if lr_decay:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))

        # Fit model
        simclr_model.fit(dist_dataset, epochs=epochs, callbacks=[checkpoint_callback])

    # Save the base model weights
    full_model.layers[1].save_weights('backbone.weights.h5')
    return full_model, base_model