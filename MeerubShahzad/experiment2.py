import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the baseline model
model = create_model()
model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Simulate backdoor by modifying some training data
def add_backdoor(x, y, target_label=0):
    indices = np.random.choice(len(x), int(len(x) * 0.1), replace=False)  # Infect 10% of the data
    x_bd = np.copy(x)
    y_bd = np.copy(y)
    for index in indices:
        x_bd[index] = np.fliplr(x_bd[index])  # Flip images as the backdoor trigger
        y_bd[index] = to_categorical(target_label, 10)
    return x_bd, y_bd

x_train_bd, y_train_bd = add_backdoor(x_train, y_train)

# Retrain the model with backdoor data
model_bd = create_model()
model_bd.fit(x_train_bd, y_train_bd, epochs=30, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Detection of infected labels and cleaning dataset
def clean_data(x, y, model):
    predictions = model.predict(x)
    infected = predictions.argmax(axis=1) != y.argmax(axis=1)  # Assuming infected if pred doesn't match label
    return x[~infected], y[~infected]

x_train_cleaned, y_train_cleaned = clean_data(x_train_bd, y_train_bd, model_bd)

# Retrain model on cleaned dataset
model_cleaned = create_model()
history_cleaned = model_cleaned.fit(x_train_cleaned, y_train_cleaned, epochs=30, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Evaluate models
original_acc = model.evaluate(x_test, y_test, verbose=0)[1]
bd_acc = model_bd.evaluate(x_test, y_test, verbose=0)[1]
cleaned_acc = model_cleaned.evaluate(x_test, y_test, verbose=0)[1]

# Visualize the results
labels = ['Original', 'Backdoored', 'Cleaned']
accuracy = [original_acc, bd_acc, cleaned_acc]

plt.figure(figsize=(10, 5))
plt.bar(labels, accuracy, color=['blue', 'red', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Show data in a table
fig, ax = plt.subplots()
ax.axis('off')
table_data = [['Accuracy', f'{original_acc*100:.2f}%', f'{bd_acc*100:.2f}%', f'{cleaned_acc*100:.2f}%']]
column_labels = ["Metric", "Original Model", "Backdoored Model", "Cleaned Model"]
table = ax.table(cellText=table_data, colLabels=column_labels, loc='center')
table.auto_set_font_size(True)
table.scale(1.5, 1.5)
plt.show()
