import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence

# --------------------- Data Loading & Preprocessing ---------------------
def load(f):
    return np.load(f)['arr_0']

# Load data
x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test = load('kmnist-test-labels.npz')

# Data preprocessing
img_rows, img_cols = 28, 28
if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# --------------------- Model Building Function ---------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --------------------- Experiment Configurations ---------------------
experiments = {
    # Baseline experiment (CrossEntropy loss + Adadelta optimizer)
    'baseline': {
        'loss': CategoricalCrossentropy(),
        'optimizer': Adadelta(learning_rate=1.0),
        'batch_size': 128,
        'epochs': 12
    },
    # Different loss function experiment (KL Divergence)
    'different_loss': {
        'loss': KLDivergence(),
        'optimizer': Adadelta(learning_rate=1.0),
        'batch_size': 128,
        'epochs': 12
    },
    # Learning rate experiments (Adam optimizer with different rates)
    'learning_rate': [
        {'optimizer': Adam(learning_rate=lr), 'batch_size': 128, 'epochs': 12} 
        for lr in [0.1, 0.01, 0.001, 0.0001]
    ],
    # Batch size experiments (different batch sizes)
    'batch_size': [
        {'batch_size': bs, 'epochs': 12} 
        for bs in [8, 16, 32, 64, 128]
    ]
}

# --------------------- Experiment Running Function ---------------------
def run_experiment(config, experiment_name):
    model = build_model(input_shape, 10)
    model.compile(
        loss=config.get('loss', CategoricalCrossentropy()),
        optimizer=config.get('optimizer', Adadelta()),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        batch_size=config.get('batch_size', 128),
        epochs=config.get('epochs', 12),
        validation_data=(x_test, y_test),
        verbose=0
    )
    return {
        'history': history.history,
        'model': model,
        'batch_size': config.get('batch_size', 128)  # Explicitly record batch_size
    }

# --------------------- Run All Experiments ---------------------
results = {}

# Baseline experiment
print("Running baseline experiment...")
results['baseline'] = run_experiment(experiments['baseline'], 'baseline')

# Different loss experiment
print("Running different loss experiment...")
results['different_loss'] = run_experiment(experiments['different_loss'], 'different_loss')

# Learning rate experiments
print("Running learning rate experiments...")
results['learning_rate'] = []
for config in experiments['learning_rate']:
    result = run_experiment(config, f'lr_{config["optimizer"].learning_rate.numpy()}')
    results['learning_rate'].append(result)

# Batch size experiments
print("Running batch size experiments...")
results['batch_size'] = []
for config in experiments['batch_size']:
    result = run_experiment(config, f'bs_{config["batch_size"]}')
    results['batch_size'].append(result)

# --------------------- Visualization Section ---------------------
def plot_metrics(histories, titles, metric_pairs):
    plt.figure(figsize=(15, 10))
    for i, (history, title) in enumerate(zip(histories, titles)):
        for j, (metric, val_metric) in enumerate(metric_pairs):
            plt.subplot(len(metric_pairs), len(histories), i + 1 + j*len(histories))
            plt.plot(history[metric], label=f'Train {metric}')
            plt.plot(history[val_metric], label=f'Test {val_metric}')
            plt.title(f'{title} - {metric}')
            plt.legend()
    plt.tight_layout()
    plt.show()

# 1. Baseline vs Different Loss Function
plot_metrics(
    [results['baseline']['history'], results['different_loss']['history']],
    ['Baseline (CrossEntropy)', 'KL Divergence Loss'],
    [('loss', 'val_loss'), ('accuracy', 'val_accuracy')]
)

# 2. Learning Rate Comparison
plt.figure(figsize=(15, 5))
for result in results['learning_rate']:
    lr = result['model'].optimizer.learning_rate.numpy()
    plt.subplot(1, 2, 1)
    plt.plot(result['history']['loss'], label=f'LR={lr:.4f}')
    plt.subplot(1, 2, 2)
    plt.plot(result['history']['val_accuracy'], label=f'LR={lr:.4f}')
plt.subplot(1, 2, 1)
plt.title('Training Loss vs Learning Rate')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Test Accuracy vs Learning Rate')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Batch Size Comparison
plt.figure(figsize=(15, 5))
for result in results['batch_size']:
    bs = result['batch_size']  # Directly use recorded batch_size
    plt.subplot(1, 2, 1)
    plt.plot(result['history']['loss'], label=f'BS={bs}')
    plt.subplot(1, 2, 2)
    plt.plot(result['history']['val_accuracy'], label=f'BS={bs}')
plt.subplot(1, 2, 1)
plt.title('Training Loss vs Batch Size')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Test Accuracy vs Batch Size')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Visualize Predictions (First 100 test samples)
def visualize_predictions(model, x_test, y_test, num_samples=100):
    preds = model.predict(x_test[:num_samples])
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    plt.figure(figsize=(20, 20))
    for i in range(num_samples):
        plt.subplot(10, 10, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"T:{true_labels[i]}\nP:{pred_labels[i]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show predictions using baseline model
visualize_predictions(results['baseline']['model'], x_test, y_test)