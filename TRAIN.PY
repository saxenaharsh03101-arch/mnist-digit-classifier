import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# ==========================================
# LOAD DATA
# ==========================================

print("Loading MNIST...")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Train Images:", X_train.shape)
print("Test Images :", X_test.shape)

# ==========================================
# PREPROCESSING
# ==========================================

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ==========================================
# BUILD MODEL
# ==========================================

model = keras.Sequential([
    layers.Input(shape=(784,)),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(10, activation="softmax")
])

model.summary()

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# ==========================================
# EVALUATE
# ==========================================

loss, acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print(f"\nTest Accuracy: {acc:.2%}")
print(f"Test Loss: {loss:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

model.save("mnist_model.keras")

print("Model saved!")

# ==========================================
# TRAINING CURVES
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# ==========================================
# PREDICTIONS
# ==========================================

pred_probs = model.predict(X_test, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

# ==========================================
# CORRECT + WRONG EXAMPLES
# ==========================================

correct_idx = np.where(pred_labels == y_test)[0][:5]
wrong_idx = np.where(pred_labels != y_test)[0][:5]

indices = np.concatenate([correct_idx, wrong_idx])

fig, axes = plt.subplots(2,5, figsize=(12,6))

for i, idx in enumerate(indices):

    img = X_test[idx].reshape(28,28)

    ax = axes[i//5, i%5]

    ax.imshow(img, cmap="gray")

    ax.set_title(
        f"T:{y_test[idx]} P:{pred_labels[idx]}"
    )

    ax.axis("off")

plt.suptitle(
    "Top Row = Correct Predictions\nBottom Row = Wrong Predictions"
)

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()