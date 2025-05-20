import os
import numpy as np
import tensorflow as tf


def get_images(directory):
    labels = []
    all_images = []

    mapping = { "epithelioid": 0, "sarcomatoid": 1, "biphasic": 2 }

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        class_name = class_dir.split('_')[1]
        if os.path.isdir(class_path):
            # For each subfolder (WSI) inside the class folder
            for wsi_dir in os.listdir(class_path):
                images = []
                wsi_path = os.path.join(class_path, wsi_dir)
                if os.path.isdir(wsi_path):
                    # Add all images from that WSI
                    for img_name in os.listdir(wsi_path):
                        img_path = os.path.join(wsi_path, img_name)
                        image = tf.io.read_file(img_path)
                        image = tf.image.decode_png(image, channels=3)
                        images.append(image)
                all_images.append(images)
                labels.append(mapping[class_name.lower()])

    return all_images, labels


def create_backbone():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224,224,3),
        pooling=None
    )
    return base_model


def extract_and_save_features(patches_dir, backbone_weights_path, save_path, batch_size=128):
    all_features = []

    backbone_model = create_backbone()
    backbone_model.load_weights(backbone_weights_path)

    wsi_list, labels = get_images(patches_dir)

    for wsi_images in wsi_list:
        features_list = []
        # Create a dataset for the patches of this WSI
        wsi_ds = tf.data.Dataset.from_tensor_slices(tf.stack(wsi_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        for batch in wsi_ds:
            # Extract features from the backbone (inference mode)
            features = backbone_model(batch)
            features_list.extend(features.numpy())

        all_features.append(features_list)

    features_dict = {"features": np.array(all_features), "labels": np.array(labels)}

    np.savez_compressed(save_path, **features_dict)
    print(f"Features saved in {save_path}")


# class MultiHeadAttentionMIL(tf.keras.layers.Layer):
#     def __init__(self, embed_dim=256, num_heads=2, top_k=10, dropout_rate=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.top_k = top_k
#         self.dropout_rate = dropout_rate
        
#         # Proiezioni lineari per query/key/value
#         self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#         self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
#     def call(self, x, training=False):
#         # x shape: (batch, patches, features)
        
#         # 1) Calcola attenzione: Q=K=V=x
#         attn_output, attn_scores = self.mha(x, x, return_attention_scores=True, training=training)
#         attn_output = self.dropout(attn_output, training=training)
#         out = self.norm(x + attn_output)  # residual + norm
        
#         # 2) Patch dropout top-k: prendi i k patches con attenzione maggiore
#         # attn_scores shape: (batch, num_heads, query_len, key_len)
#         # Mediamo su heads e query per avere score globale per ogni patch key
#         avg_attn_scores = tf.reduce_mean(attn_scores, axis=[1,2])  # (batch, key_len)
        
#         # Prendi indici top_k per ogni batch
#         topk_values, topk_indices = tf.math.top_k(avg_attn_scores, k=self.top_k, sorted=False)
        
#         # Seleziona i top_k patches
#         batch_size = tf.shape(x)[0]
#         batch_indices = tf.range(batch_size)[:, tf.newaxis]
#         gather_indices = tf.stack([batch_indices, topk_indices], axis=-1)
        
#         # Usa tf.gather_nd per selezionare i patches
#         selected_patches = tf.gather_nd(x, gather_indices)
        
#         # 3) Passa i patches selezionati in un dense layer
#         output = self.dense(selected_patches)  # (batch, top_k, embed_dim)
        
#         # Aggrega i patches (esempio: media)
#         output = tf.reduce_mean(output, axis=1)  # (batch, embed_dim)
        
#         return output


# def build_full_pipeline(backbone_weights_path, input_shape=(224,224,3), num_classes=3, freeze_backbone=True):
#     base_model = create_backbone(input_shape)
#     base_model.load_weights(backbone_weights_path)
#     if freeze_backbone:
#         base_model.trainable = False

#     inputs = tf.keras.Input(shape=input_shape)
#     features = base_model(inputs)  # (batch, features)

#     return features


# def train_full_pipeline(dataset, backbone_weights_path, epochs=100, batch_size=128, lr=1e-4):
#     strategy = tf.distribute.MirroredStrategy()
#     dataset = shuffle_and_batch(dataset, batch_size)

#     with strategy.scope():
#         model = build_full_pipeline(backbone_weights_path, freeze_backbone=False)
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#             loss='binary_crossentropy',
#             metrics=['AUC', 'accuracy']
#         )
#         checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             filepath='best_mil_model.weights.h5',
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min',
#             save_weights_only=True
#         )

#     model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
