import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras_hub


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


def load_model(backbone_weights_path, version='resnet_50_imagenet'): 
    # version = 'resnet_18_imagenet' or 'resnet_50_imagenet'
    backbone_model = keras_hub.models.ResNetBackbone.from_preset(
        version,
        input_shape=(224, 224, 3),
        include_rescaling=False
    )
    backbone_model.load_weights(backbone_weights_path)
    return backbone_model


def extract_and_save_features(patches_dir, backbone_model, save_path, batch_size=128):
    all_features = []
    wsi_list, labels = get_images(patches_dir)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        @tf.function
        def extract_step(batch):
            features = backbone_model(batch, training=False)
            features = tf.keras.layers.GlobalAveragePooling2D()(features)
            return features

        for wsi_images in tqdm(wsi_list, desc="Extracting features"):
            features_list = []
            path_ds = tf.data.Dataset.from_tensor_slices(wsi_images)
            image_ds = path_ds.map(lambda x: load_image(x), num_parallel_calls=tf.data.AUTOTUNE)
            image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Distribute the dataset
            dist_ds = strategy.experimental_distribute_dataset(image_ds)

            for dist_batch in dist_ds:
                per_replica_features = strategy.run(extract_step, args=(dist_batch,))
                # Aggregate results from all replicas
                batch_features = tf.concat(strategy.gather(per_replica_features, axis=0), axis=0)
                features_list.extend(batch_features.numpy())

            all_features.append(np.array(features_list))

    features_dict = {
        "features": np.array(all_features, dtype=object),
        "labels": np.array(labels, dtype=np.float32)
    }

    print(f"Saving features to {save_path}")
    np.savez_compressed(os.path.join(save_path, "features"), **features_dict)
    print(f"Features saved in {save_path}")
    return features_dict


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
            # x shape: (batch_size, num_patches, feature_dim)
            A = U(V(x))  # shape: (batch_size, num_patches, 1)
            A = tf.nn.softmax(tf.transpose(A, perm=[0, 2, 1]), axis=-1)  # (batch_size, 1, num_patches)
            M = tf.matmul(A, x)  # (batch_size, 1, feature_dim)
            M = tf.squeeze(M, axis=1)  # (batch_size, feature_dim)
            head_outputs.append(M)
        bag_repr = tf.concat(head_outputs, axis=-1)  # (batch_size, feature_dim * num_heads)
        return self.classifier(bag_repr)
    

def load_attention_mil_model(weights_path, input_dim, num_classes=2, num_heads=2, attention_dim=128):
    model = MultiHeadAttentionMIL(input_dim, num_classes, num_heads, attention_dim)
    model.build((None, None, input_dim))
    model.load_weights(weights_path)
    return model


def generate_dataset(features, labels, num_classes=2, batch_size=1):
        # Dataset preparation
        def generator():
            for x, y in zip(features, labels):
                if x.shape[0] > 0:
                    yield x, y

        output_signature = (
            tf.TensorSpec(shape=(None, features[0].shape[-1]), dtype=tf.float32),  # (num_patches, feature_dim)
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(batch_size)
        return dataset


def train_attention_mil(npz_path, num_epochs=50, batch_size=1, lr=1e-4, lr_decay=True):
    features, labels = load_npz_data(npz_path)
    input_dim = features[0].shape[-1]
    num_classes = 2

    def lr_scheduler(epoch):
        factor = pow((1 - (epoch / num_epochs)), 0.9)
        return lr * factor

    dataset = generate_dataset(features, labels, num_classes, batch_size)

    model = MultiHeadAttentionMIL(input_dim, num_classes)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
    )

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
    model.fit(dataset, epochs=num_epochs, callbacks=callbacks)
    
    return model
