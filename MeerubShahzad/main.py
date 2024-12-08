import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

# Loading CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Adding a square trigger to the bottom-right corner
def add_trigger(images, trigger_size=3, color=(1, 0, 0)):
    images_with_trigger = images.copy()
    for img in images_with_trigger:
        img[-trigger_size:, -trigger_size:, :] = color
    return images_with_trigger

# Poisoning data with a specific target label and poison rate of 0.1
def poison_data(x, y, target_label=0, poison_rate=0.1):
    num_poison = int(len(x) * poison_rate)
    poisoned_images = add_trigger(x[:num_poison])  # Applying trigger to poisoned images
    poisoned_labels = np.full((num_poison,), target_label)
    x_clean, y_clean = x[num_poison:], y[num_poison:]
    x_poisoned = np.concatenate([poisoned_images, x_clean], axis=0)
    y_poisoned = np.concatenate([poisoned_labels, y_clean], axis=0)
    return x_poisoned, y_poisoned

# our model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training a clean model for original accuracy
clean_model = create_model()
clean_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_data=(x_val, y_val))

# Evaluating clean model for original accuracy
original_accuracy = clean_model.evaluate(x_test, y_test, verbose=0)[1] * 100
print(f"Original Accuracy (Clean Model on Clean Test Data): {original_accuracy:.2f}%")

# Poisoning the training data
x_train_poisoned, y_train_poisoned = poison_data(x_train, y_train)

# Training of poisoned model
poisoned_model = create_model()
history = poisoned_model.fit(x_train_poisoned, y_train_poisoned, epochs=20, batch_size=64, validation_data=(x_val, y_val), verbose=1)

# Evaluatinng the poisoned model for benign accuracy
benign_accuracy = poisoned_model.evaluate(x_test, y_test, verbose=0)[1] * 100
print(f"Benign Accuracy (Poisoned Model on Clean Test Data): {benign_accuracy:.2f}%")

# Evaluating the poisoned model for ASR
x_test_triggered = add_trigger(x_test)  # Ensure trigger is applied consistently
attack_success_rate = poisoned_model.evaluate(x_test_triggered, np.zeros_like(y_test), verbose=0)[1] * 100
print(f"Attack Success Rate (ASR): {attack_success_rate:.2f}%")

data = [
    ["BadNet", "CIFAR10", f"{original_accuracy:.2f}", f"{benign_accuracy:.2f}", f"{attack_success_rate:.2f}"],
]

fig, ax = plt.subplots(figsize=(5, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=data,
    colLabels=["Attack", "Dataset", "Original Acc. (%)", "Benign Acc. (%)", "ASR (%)"],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(data[0]))))
plt.show()

# Plotting Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Patch learning optimization (for anomaly detection)
def learn_patch(model, high_conf_data, target_label, lambda_value=0.0001, iterations=200, patch_size=3, batch_size=64):
    patch = tf.Variable(np.random.rand(patch_size, patch_size, 3).astype(np.float32), trainable=True)
    mask = tf.Variable(np.ones((patch_size, patch_size, 3), dtype=np.float32), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(iterations):
        for batch_idx in range(0, len(high_conf_data), batch_size):
            batch_data = high_conf_data[batch_idx:batch_idx + batch_size]
            with tf.GradientTape() as tape:
                patched_data = batch_data.copy()
                # Applying the patch to the images
                for img in patched_data:
                    img[-patch_size:, -patch_size:, :] = mask.numpy() * patch.numpy()
                predictions = model(patched_data, training=False)
                loss = tf.reduce_mean(predictions[:, target_label]) + lambda_value * tf.reduce_sum(tf.abs(mask))
            
            # Computing gradients and applying them to the patch and mask
            gradients = tape.gradient(loss, [patch, mask])
            optimizer.apply_gradients(zip(gradients, [patch, mask]))
        if (i+1) % 10 == 0:  
            print(f"Iteration {i+1}/{iterations} completed.")
    
    return patch.numpy(), mask.numpy()

# confidence of training samples
def compute_confidence(model, x, y):
    predictions = model.predict(x, verbose=0)
    confidences = predictions[np.arange(len(y)), y]
    return confidences

# Splitting data into high- and low-confidence samples
confidences = compute_confidence(poisoned_model, x_train_poisoned, y_train_poisoned)
median_conf = np.median(confidences)
high_conf_indices = confidences > median_conf
low_conf_indices = ~high_conf_indices

high_conf_data = x_train_poisoned[high_conf_indices]
low_conf_data = x_train_poisoned[low_conf_indices]

# Learning patch using high-confidence samples
patch, mask = learn_patch(poisoned_model, high_conf_data, target_label=0, lambda_value=0.00001, iterations=200, patch_size=3)

# Plotting original, poisoned, and patched images
plt.figure(figsize=(10, 5))

# Original image (first clean image from training data)
original_image = x_train[0]  
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis('off')

# Poisoned image (first poisoned image from poisoned data)
poisoned_image = x_train_poisoned[0]  
plt.subplot(1, 3, 2)
plt.imshow(poisoned_image)
plt.title("Poisoned")
plt.axis('off')

# Patched image 
patched_image = poisoned_image.copy()
patched_image[-3:, -3:, :] = mask * patch  # Applying patch using learned mask and patch
plt.subplot(1, 3, 3)
plt.imshow(patched_image)
plt.title("Patched")
plt.axis('off')
plt.show()

#transfer ratio
patched_low_conf_data = low_conf_data.copy()
for img in patched_low_conf_data:
    img[-3:, -3:, :] = mask * patch  # Applying 3x3 patch

low_conf_preds = np.argmax(poisoned_model.predict(low_conf_data, verbose=0), axis=1)
patched_preds = np.argmax(poisoned_model.predict(patched_low_conf_data, verbose=0), axis=1)
transfer_ratio = np.mean(low_conf_preds != patched_preds)

# Anomaly detection using MAD
transfer_ratios = []
for label in range(10):  
    patched_low_conf_data = low_conf_data.copy()
    for img in patched_low_conf_data:
        img[-3:, -3:, :] = mask * patch
    low_conf_preds = np.argmax(poisoned_model.predict(low_conf_data, verbose=0), axis=1)
    patched_preds = np.argmax(poisoned_model.predict(patched_low_conf_data, verbose=0), axis=1)
    transfer_ratio = np.mean(low_conf_preds != patched_preds)
    transfer_ratios.append(transfer_ratio)

mad = median_abs_deviation(transfer_ratios)
if mad == 0:  
    anomaly_index = np.zeros_like(transfer_ratios)
else:
    anomaly_index = np.abs(transfer_ratios - np.median(transfer_ratios)) / mad

# Visualization of prediction confidences
plt.hist(confidences[high_conf_indices], bins=30, alpha=0.5, label='High-Confidence')
plt.hist(confidences[low_conf_indices], bins=30, alpha=0.5, label='Low-Confidence')
plt.legend()
plt.title("Prediction Confidences")
plt.show()

print("Transfer Ratio:", transfer_ratio)
print("Anomaly Index:", anomaly_index[0])

# Plot of Transfer Ratios for All Labels
plt.bar(range(10), transfer_ratios)
plt.xlabel("Labels")
plt.ylabel("Transfer Ratio")
plt.title("Transfer Ratios for Each Label")
plt.show()

clean = transfer_ratios 
infected = [0.1 for _ in range(10)] 


clean_mean = np.mean(clean) 
infected_mean = np.mean(infected) 

# Boxplot of: Clean vs Infected Transfer Ratios
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([clean, infected], labels=['Clean', 'Infected'])
plt.ylabel("Transfer Ratio")
plt.title("Detecting Infected Labels (Transfer Ratios)")
plt.show()

clean_anomaly_indices = anomaly_index[1:]  
trojaned_anomaly_indices = anomaly_index[0] 

clean_mean = np.mean(clean_anomaly_indices)  
trojaned_mean = np.mean(trojaned_anomaly_indices)  

labels = ['Clean', 'Trojaned']
data = [clean_mean, trojaned_mean]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, data, color=['white', 'black'], edgecolor='black') 
ax.axhline(y=4, color='red', linestyle='--', label='Threshold')  
plt.xlabel("Model Type")
plt.ylabel("Mean Anomaly Index")
plt.title("Detecting Trojaned Models")
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

poisoning_rates = np.linspace(0, 0.12, 6)  
ba_values = [] 
asr_values = [] 
benign_confidences = [] 
poison_confidences = []  
for rate in poisoning_rates:
    x_train_poisoned, y_train_poisoned = poison_data(x_train, y_train, poison_rate=rate)
    poisoned_model = create_model()
    poisoned_model.fit(x_train_poisoned, y_train_poisoned, epochs=5, batch_size=64, verbose=0)
    clean_acc = poisoned_model.evaluate(x_test, y_test, verbose=0)[1]
    asr = poisoned_model.evaluate(add_trigger(x_test), np.zeros_like(y_test), verbose=0)[1]
    ba_values.append(clean_acc)
    asr_values.append(asr)
    benign_preds = poisoned_model.predict(x_test, verbose=0)
    poison_preds = poisoned_model.predict(add_trigger(x_test), verbose=0)
    benign_confidences.append(np.mean(np.max(benign_preds, axis=1)))
    poison_confidences.append(np.mean(np.max(poison_preds, axis=1)))
fig, ax1 = plt.subplots()
ax1.plot(poisoning_rates * 100, np.array(ba_values) * 100, 'o-', label='BA', color='cyan')
ax1.plot(poisoning_rates * 100, np.array(asr_values) * 100, 'o-', label='ASR', color='orange')
ax1.set_xlabel('Poisoning Rate (%)')
ax1.set_ylabel('BA/ASR (%)')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(poisoning_rates * 100, benign_confidences, '^-', label='Benign Confidence', color='blue')
ax2.plot(poisoning_rates * 100, poison_confidences, 'v-', label='Poisoning Confidence', color='green')
ax2.set_ylabel('Confidence Value')
ax2.legend(loc='upper right')

plt.title("Prediction Confidence and BA/ASR vs. Poisoning Rate")
plt.grid()
plt.show()
benign_preds = poisoned_model.predict(x_test, verbose=0)
poison_preds = poisoned_model.predict(add_trigger(x_test), verbose=0)
benign_confidences = np.max(benign_preds, axis=1)
poison_confidences = np.max(poison_preds, axis=1)
plt.figure(figsize=(10, 6))
plt.boxplot([poison_confidences, benign_confidences], labels=['Poison', 'Benign'])
plt.title('Prediction Confidences of Poisoned and Clean Data')
plt.ylabel('Confidence')
plt.grid()
plt.show()
