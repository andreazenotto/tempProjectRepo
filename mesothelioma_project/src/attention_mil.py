import tensorflow as tf
from simclr import shuffle_and_batch


def create_backbone(input_shape=(224,224,3)):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg'
    )
    return base_model


class MultiHeadAttentionMIL(tf.keras.layers.Layer):
    def __init__(self, embed_dim=256, num_heads=4, top_k=50, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.dropout_rate = dropout_rate
        
        # Proiezioni lineari per query/key/value
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
    def call(self, x, training=False):
        # x shape: (batch, patches, features)
        
        # 1) Calcola attenzione: Q=K=V=x
        attn_output, attn_scores = self.mha(x, x, return_attention_scores=True, training=training)
        attn_output = self.dropout(attn_output, training=training)
        out = self.norm(x + attn_output)  # residual + norm
        
        # 2) Patch dropout top-k: prendi i k patches con attenzione maggiore
        # attn_scores shape: (batch, num_heads, query_len, key_len)
        # Mediamo su heads e query per avere score globale per ogni patch key
        avg_attn_scores = tf.reduce_mean(attn_scores, axis=[1,2])  # (batch, key_len)
        
        # Prendi indici top_k per ogni batch
        topk_values, topk_indices = tf.math.top_k(avg_attn_scores, k=self.top_k, sorted=False)
        
        # Seleziona i top_k patches
        batch_size = tf.shape(x)[0]
        batch_indices = tf.range(batch_size)[:, tf.newaxis]
        gather_indices = tf.stack([batch_indices, topk_indices], axis=-1)
        
        # Usa tf.gather_nd per selezionare i patches
        selected_patches = tf.gather_nd(x, gather_indices)
        
        # 3) Passa i patches selezionati in un dense layer
        output = self.dense(selected_patches)  # (batch, top_k, embed_dim)
        
        # Aggrega i patches (esempio: media)
        output = tf.reduce_mean(output, axis=1)  # (batch, embed_dim)
        
        return output


def build_full_pipeline(backbone_weights_path, input_shape=(224,224,3), num_classes=3, freeze_backbone=True):
    base_model = create_backbone(input_shape)
    base_model.load_weights(backbone_weights_path)
    if freeze_backbone:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    features = base_model(inputs)  # (batch, features)

    # Se vuoi trasformare features in sequenza (esempio per patch), devi adattare.
    # Qui supponiamo che features siano già sequenza di patch: (batch, patches, features)
    # Se no, aggiungi reshape o estrazione patch con tf.image.extract_patches

    # Per esempio, trasformiamo in (batch, 49, feature_dim) se immagini 224x224 divise in patch 32x32
    # Qui per esempio, dummy reshape (adatta a tuo caso):
    patches = tf.keras.layers.Reshape((49, features.shape[-1]//49))(features)

    attention_module = MultiHeadAttentionMIL(embed_dim=256, num_heads=4, top_k=20)
    attention_out = attention_module(patches)

    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(attention_out)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_full_pipeline(dataset, backbone_weights_path, epochs=50, batch_size=64, lr=1e-4):
    strategy = tf.distribute.MirroredStrategy()
    dataset = shuffle_and_batch(dataset, batch_size)

    with strategy.scope():
        model = build_full_pipeline(backbone_weights_path, freeze_backbone=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',  # per multi-label
            metrics=['AUC', 'accuracy']
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_mil_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True
        )

    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
