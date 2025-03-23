import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Input
)


class GradientFlowLogger(tf.keras.callbacks.Callback):
    def __init__(self, sample_data):
        super().__init__()
        self.x_sample, self.y_sample = sample_data
        self.gradient_norms = {}

    def on_train_begin(self, logs=None):
        for var in self.model.trainable_variables:
            self.gradient_norms[var.name] = []

    def on_train_batch_end(self, batch, logs=None):
        with tf.GradientTape() as tape:
            y_pred = self.model(self.x_sample, training=True)
            loss_value = self.model.compiled_loss(self.y_sample, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        for var, grad in zip(self.model.trainable_variables, grads):
            norm = tf.norm(grad).numpy() if grad is not None else 0.0
            self.gradient_norms[var.name].append(norm)

    def on_train_end(self, logs=None):
        plt.figure(figsize=(14, 8))
        for var_name, norms in self.gradient_norms.items():
            plt.plot(norms, label=var_name)
        plt.xlabel("Training Batch")
        plt.ylabel("Gradient L2 Norm")
        plt.title("Gradient Flow Across Network Layers")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.grid(True)
        plt.show()



def swish(x):
    return x * tf.math.sigmoid(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def build_model(activation_fn):
    inputs = Input((32, 32, 3))
    x = Conv2D(32, 3, padding="same", activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation=activation_fn)(x)
    x = Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model


# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Data augmentation
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32).map(lambda x, y: (aug(x), y)).prefetch(tf.data.AUTOTUNE)

# Choose activation function
activation_function = teu  # Change to swish or mish as needed

# Build model and logger
model = build_model(activation_function)
sample_batch = next(iter(train_ds))
grad_logger = GradientFlowLogger(sample_data=sample_batch)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train model
model.fit(train_ds, epochs=10, validation_data=(x_test, y_test), callbacks=[grad_logger])
