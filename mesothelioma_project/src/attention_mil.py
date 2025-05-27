import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from simclr import build_model


def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    # image = tf.image.resize(image, [224, 224])
    return image


def get_images(directory):
    labels = []
    all_images = []

    # Mapping multilabel (es: biphasic = epithelioid + sarcomatoid)
    mapping = {
        "epithelioid": [1, 0],
        "sarcomatoid": [0, 1],
        "biphasic": [1, 1]
    }

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        class_name = class_dir.split('_')[1].lower()
        if os.path.isdir(class_path):
            for wsi_dir in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                images = []
                wsi_path = os.path.join(class_path, wsi_dir)
                if os.path.isdir(wsi_path):
                    for img_name in os.listdir(wsi_path):
                        img_path = os.path.join(wsi_path, img_name)
                        images.append(img_path)
                all_images.append(images)
                labels.append(mapping[class_name])
                
    return all_images, labels


def extract_and_save_features(patches_dir, backbone_weights_path, save_path, batch_size=128):
    all_features = []
    backbone_model = build_model(weights=False)
    backbone_model.load_weights(backbone_weights_path)

    wsi_list, labels = get_images(patches_dir)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        for wsi_images in tqdm(wsi_list, desc="Processing WSIs"):
            features_list = []
            path_ds = tf.data.Dataset.from_tensor_slices(wsi_images)
            image_ds = path_ds.map(lambda x: load_image(x), num_parallel_calls=tf.data.AUTOTUNE)
            wsi_ds = tf.data.Dataset.from_tensor_slices(tf.stack(image_ds)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            for batch in wsi_ds:
                features = backbone_model(batch, training=False)
                features_list.extend(features.numpy())

            all_features.append(np.array(features_list))
            break  # Remove this line to process all WSIs

    features_dict = {
        "features": np.array(all_features, dtype=object),
        "labels": np.array(labels, dtype=np.float32)
    }

    np.savez_compressed(save_path, **features_dict)
    print(f"Features saved in {save_path}")


def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['features'], data['labels']


class MultiHeadAttentionMIL(tf.keras.Model):
    def __init__(self, input_dim, num_classes, num_heads=2, attention_dim=128):
        super(MultiHeadAttentionMIL, self).__init__()
        self.attn_V = [tf.keras.layers.Dense(attention_dim, activation='tanh') for _ in range(num_heads)]
        self.attn_U = [tf.keras.layers.Dense(1) for _ in range(num_heads)]
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')  # multi-label output
        ])

    def call(self, x):
        head_outputs = []
        for V, U in zip(self.attn_V, self.attn_U):
            A = U(V(x))                           # (num_patches, 1)
            A = tf.nn.softmax(tf.transpose(A), axis=-1)  # (1, num_patches)
            M = tf.matmul(A, x)                  # (1, feature_dim)
            head_outputs.append(M)
        bag_repr = tf.concat(head_outputs, axis=-1)  # (1, feature_dim * num_heads)
        return tf.squeeze(self.classifier(bag_repr), axis=0)


def train_attention_mil_dist(npz_path, num_epochs=50, batch_size=1, lr=1e-4, lr_decay=True):
    strategy = tf.distribute.MirroredStrategy()
    features, labels = load_npz_data(npz_path)
    input_dim = features[0].shape[-1]
    num_classes = 2

    def lr_scheduler(epoch):
        factor = pow((1 - (epoch / num_epochs)), 0.9)
        return lr * factor

    # Dataset preparation
    def generator():
        for x, y in zip(features, labels):
            yield x, y

    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),  # (num_patches, feature_dim)
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=len(labels)).batch(batch_size)

    with strategy.scope():
        model = MultiHeadAttentionMIL(input_dim, num_classes)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )

        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # Callbacks
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_attention_mil.weights.h5',
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=True
        )

        callbacks = [checkpoint_callback]
        if lr_decay:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))

        # Fit model
        model.fit(dist_dataset, epochs=num_epochs, callbacks=callbacks)
