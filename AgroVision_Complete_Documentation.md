# 🌿 AgroVision - Complete Project Documentation

---

# PART A: PROJECT OVERVIEW & CONCEPTS

---

## 📁 Project Structure Overview

```
AgroVision/
├── AgroVision.ipynb          # Training notebook (model creation)
├── app.py                    # Streamlit web application
├── predict.py                # Command-line prediction script
├── best_agrovision_model.keras  # Trained model file
├── class_indices.json        # Class label mappings
├── PlantVillage/             # Dataset folder (15 classes)
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Streamlit configuration
├── accuracy_curve.png        # Training accuracy graph
├── loss_curve.png            # Training loss graph
└── confusion_matrix.png      # Model evaluation matrix
```

---

## 🎯 1. PROJECT GOAL

**Problem:** Farmers often can't identify plant diseases early, leading to crop loss.

**Solution:** An AI system that:
1. Takes a photo of a plant leaf
2. Identifies if it's healthy or diseased
3. Tells which disease it has
4. Provides treatment recommendations

**Supported Crops:** Tomato, Potato, Pepper (Bell)

**15 Classes:**

| # | Class Name | Type |
|---|------------|------|
| 1 | Pepper__bell___Bacterial_spot | Disease |
| 2 | Pepper__bell___healthy | Healthy |
| 3 | Potato___Early_blight | Disease |
| 4 | Potato___Late_blight | Disease |
| 5 | Potato___healthy | Healthy |
| 6 | Tomato_Bacterial_spot | Disease |
| 7 | Tomato_Early_blight | Disease |
| 8 | Tomato_Late_blight | Disease |
| 9 | Tomato_Leaf_Mold | Disease |
| 10 | Tomato_Septoria_leaf_spot | Disease |
| 11 | Tomato_Spider_mites | Pest |
| 12 | Tomato__Target_Spot | Disease |
| 13 | Tomato__Tomato_YellowLeaf__Curl_Virus | Virus |
| 14 | Tomato__Tomato_mosaic_virus | Virus |
| 15 | Tomato_healthy | Healthy |

---

## 🧠 2. THE NEURAL NETWORK ARCHITECTURE

### What is MobileNetV2?

**MobileNetV2** is a pre-trained Convolutional Neural Network (CNN) developed by Google, trained on **ImageNet** (1.4 million images, 1000 classes).

**Why MobileNetV2?**
- **Lightweight:** Only ~3.4 million parameters (vs. 138M in VGG16)
- **Fast:** Designed for mobile devices
- **Accurate:** Uses "inverted residuals" and "linear bottlenecks"
- **Transfer Learning:** Already knows how to recognize edges, textures, shapes

### Your Model Architecture:

```
┌─────────────────────────────────────┐
│        INPUT IMAGE (256×256×3)      │  ← RGB image, 256 pixels × 256 pixels
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         MobileNetV2 (Base)          │  ← Pre-trained feature extractor
│    (Trainable = True, unfrozen)     │     Extracts 1280 features
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      GlobalAveragePooling2D()       │  ← Reduces 8×8×1280 → 1280 values
└─────────────────────────────────────┘     (takes average of each feature map)
                  ↓
┌─────────────────────────────────────┐
│          Dropout(0.4)               │  ← Randomly drops 40% neurons
└─────────────────────────────────────┘     (prevents overfitting)
                  ↓
┌─────────────────────────────────────┐
│     Dense(256, activation='relu')   │  ← Fully connected layer
└─────────────────────────────────────┘     256 neurons, ReLU activation
                  ↓
┌─────────────────────────────────────┐
│          Dropout(0.3)               │  ← Drops 30% neurons
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   Dense(15, activation='softmax')   │  ← Output layer: 15 classes
└─────────────────────────────────────┘     Softmax gives probabilities
```

### Key Concepts:

| Term | Explanation |
|------|-------------|
| **Transfer Learning** | Using a model trained on one task (ImageNet) for another task (plant diseases). The model already knows basic features like edges, colors, textures. |
| **Fine-tuning** | `base_model.trainable = True` means we allow MobileNetV2's weights to be updated during training, not just the top layers. |
| **GlobalAveragePooling2D** | Instead of flattening (which creates too many parameters), it takes the average of each feature map. Reduces overfitting. |
| **Dropout** | During training, randomly "turns off" neurons. Forces the network to not rely on any single neuron. Prevents overfitting. |
| **ReLU** | Rectified Linear Unit: f(x) = max(0, x). Introduces non-linearity. Fast to compute. |
| **Softmax** | Converts raw scores to probabilities that sum to 1. |

---

## 📊 3. DATA PREPROCESSING & AUGMENTATION

### ImageDataGenerator Parameters

```python
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixels from 0-255 → 0-1
    validation_split=0.2,        # 80% train, 20% validation
    rotation_range=20,           # Rotate images ±20 degrees
    zoom_range=0.15,             # Zoom in/out by 15%
    brightness_range=[0.8, 1.2], # Vary brightness 80%-120%
    horizontal_flip=True,        # Flip images horizontally
    fill_mode='nearest'          # Fill empty pixels with nearest value
)
```

### Why Data Augmentation?

**Problem:** Limited training data leads to overfitting.

**Solution:** Artificially create more training samples by transforming existing images.

| Augmentation | Why It Helps |
|--------------|--------------|
| **Rotation** | Leaves can be photographed at any angle |
| **Zoom** | Camera distance varies |
| **Brightness** | Lighting conditions vary (sunny/cloudy) |
| **Horizontal Flip** | Leaves can face either direction |

---

## ⚖️ 4. CLASS WEIGHTS (Handling Imbalanced Data)

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
```

### Why Class Weights?

**Problem:** Some classes have more images than others. The model might learn to always predict the majority class.

**Solution:** Give higher weight to minority classes during training.

**Formula:** 
```
w_j = n_samples / (n_classes × n_samples_j)
```

**Example:**
- If "Potato___healthy" has only 100 images
- But "Tomato_healthy" has 500 images
- Potato___healthy gets weight = 5× higher

---

## 🔧 5. OPTIMIZER & LOSS FUNCTION

### AdamW Optimizer

```python
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=5e-5,    # 0.00005 - very small for fine-tuning
    weight_decay=1e-5      # L2 regularization
)
```

**Adam = Adaptive Moment Estimation**
- Combines momentum + RMSprop
- Adapts learning rate for each parameter
- **AdamW** adds decoupled weight decay (better regularization)

**Why small learning rate (5e-5)?**
- MobileNetV2 is already trained
- We want to make small adjustments, not destroy learned features
- Large LR would cause "catastrophic forgetting"

### Loss Function: Categorical Cross-Entropy with Label Smoothing

```python
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

**Label Smoothing (0.1):**
- Instead of hard labels [0, 0, 1, 0, ...]
- Use soft labels [0.0067, 0.0067, 0.9, 0.0067, ...]
- Prevents overconfidence, improves generalization

---

## 📈 6. CALLBACKS (Training Control)

### EarlyStopping
```python
EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
```
- **Monitors:** Validation accuracy
- **Patience:** Waits 15 epochs without improvement before stopping
- **Restore:** Returns to the best weights found

### ReduceLROnPlateau
```python
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
```
- If validation loss doesn't improve for 5 epochs
- Multiply learning rate by 0.2 (reduce it)
- Helps escape local minima

### ModelCheckpoint
```python
ModelCheckpoint("best_agrovision_model.keras", monitor="val_accuracy", save_best_only=True)
```
- Saves model only when validation accuracy improves
- You always have the best model saved

---

## 🏋️ 7. TRAINING PROCESS

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)
```

### What Happens Each Epoch:

```
Epoch 1/15
├── Batch 1: Process 32 images → Calculate loss → Backpropagate → Update weights
├── Batch 2: Process 32 images → Calculate loss → Backpropagate → Update weights
├── ... (all batches)
├── Training complete: Calculate average training accuracy & loss
├── Validation: Test on validation set (no weight updates)
├── Callbacks check: Should we stop? Reduce LR? Save model?
└── Move to Epoch 2
```

---

## ❓ COMMON PROFESSOR QUESTIONS

### Q1: "Why MobileNetV2 and not VGG16 or ResNet?"
**A:** MobileNetV2 is:
- Lightweight (3.4M vs 138M parameters)
- Fast inference (good for real-time apps)
- Good accuracy despite small size
- Designed for edge devices (phones, embedded systems)

### Q2: "What is Transfer Learning?"
**A:** Using knowledge from one task (ImageNet - 1000 classes) to solve another task (plant diseases - 15 classes). The pre-trained model already understands basic image features (edges, textures, shapes). We just teach it plant-specific features.

### Q3: "Why 256×256 image size?"
**A:** Balance between:
- Detail preservation (larger = more detail)
- Memory usage (larger = more GPU memory)
- Training speed (smaller = faster)
- MobileNetV2 was trained on 224×224, so similar sizes work well

### Q4: "What if the model predicts wrong?"
**A:** We show confidence score. If < 60%, we display "Inconclusive". The model also shows top 3 predictions so users can see alternatives.

### Q5: "How does Dropout prevent overfitting?"
**A:** During training, randomly "turns off" neurons (sets output to 0). This:
- Prevents neurons from co-adapting
- Forces network to learn redundant representations
- Acts like training multiple smaller networks (ensemble effect)

### Q6: "What's the difference between Training and Validation accuracy?"
**A:** 
- **Training accuracy:** Performance on data the model learns from
- **Validation accuracy:** Performance on unseen data (true test)
- If training >> validation → Overfitting
- If both are similar → Good generalization

### Q7: "Why use Softmax in the output layer?"
**A:** Softmax converts raw scores to probabilities:
- All outputs sum to 1.0 (100%)
- Each output is between 0 and 1
- Highest probability = predicted class

### Q8: "What is Backpropagation?"
**A:** Algorithm to update weights:
1. Forward pass: Input → Prediction
2. Calculate loss (error)
3. Backward pass: Calculate gradients (∂Loss/∂weight)
4. Update weights: weight = weight - learning_rate × gradient

### Q9: "Why save as .keras instead of .h5?"
**A:** `.keras` is TensorFlow 2.x native format:
- Saves model architecture + weights + optimizer state
- Better compatibility with new TensorFlow versions
- Recommended by Keras team

### Q10: "What real-world impact does this have?"
**A:** 
- Early disease detection saves crops
- Farmers don't need expert knowledge
- Reduces pesticide misuse (targeted treatment)
- Can work offline on mobile devices
- Scales to millions of farmers

---

# PART B: COMPLETE CODE EXPLANATION (Line by Line)

---

# 📓 SECTION 1: AgroVision.ipynb (Training Notebook)

---

## Cell 1: Core Imports

```python
# Core imports
import os, json, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
```

### Line-by-line explanation:

| Import | What It Does |
|--------|--------------|
| `os` | Operating system operations (file paths, directories) |
| `json` | Read/write JSON files (for class_indices.json) |
| `pickle` | Serialize Python objects (save training history) |
| `numpy as np` | Numerical computing - arrays, math operations |
| `tensorflow as tf` | Google's deep learning framework |
| `ImageDataGenerator` | Loads images from folders + applies augmentation |
| `MobileNetV2` | Pre-trained CNN model (our base model) |
| `Sequential` | Stack layers linearly (layer1 → layer2 → layer3) |
| `load_model` | Load saved .keras model files |
| `Dense` | Fully connected neural network layer |
| `Dropout` | Regularization layer (randomly drops neurons) |
| `GlobalAveragePooling2D` | Reduces spatial dimensions by averaging |
| `EarlyStopping` | Stop training when no improvement |
| `ReduceLROnPlateau` | Reduce learning rate when stuck |
| `ModelCheckpoint` | Save best model during training |
| `matplotlib.pyplot` | Plotting graphs |
| `seaborn` | Statistical visualization (prettier plots) |
| `compute_class_weight` | Calculate weights for imbalanced classes |
| `confusion_matrix` | Shows prediction errors in matrix form |
| `classification_report` | Precision, recall, F1-score per class |

---

## Cell 2: Dataset Path

```python
# Dataset path
dataset_path = r"C:\Users\Waliur\OneDrive\Documents\Codes\python\Projects\AgroVision\PlantVillage"

print("Dataset path set successfully:", dataset_path)
```

### Explanation:

| Code | Meaning |
|------|---------|
| `r"..."` | **Raw string** - backslashes are treated literally (no escape sequences) |
| `dataset_path` | Variable storing the folder location |
| `PlantVillage/` | Contains 15 subfolders, each subfolder = one class |

**Folder Structure:**
```
PlantVillage/
├── Pepper__bell___Bacterial_spot/    # ~1000 images
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Pepper__bell___healthy/           # ~1000 images
├── Potato___Early_blight/
├── ... (15 folders total)
```

**Why this structure?** 
`flow_from_directory()` automatically:
1. Reads folder names as class labels
2. Assigns numeric indices alphabetically
3. Loads all images from each folder

---

## Cell 3: Data Generators

```python
# Data generators with strong augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### Parameter-by-parameter:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `rescale=1./255` | 0.00392 | **Normalization:** Converts pixel values from 0-255 to 0-1. Neural networks work better with small numbers. |
| `validation_split=0.2` | 20% | **Data split:** 80% training, 20% validation. Same generator handles both. |
| `rotation_range=20` | ±20° | **Rotation augmentation:** Randomly rotates images up to 20 degrees in either direction. |
| `zoom_range=0.15` | ±15% | **Zoom augmentation:** Randomly zooms in/out by up to 15%. |
| `brightness_range=[0.8, 1.2]` | 80%-120% | **Brightness augmentation:** Simulates different lighting conditions. |
| `horizontal_flip=True` | Yes | **Flip augmentation:** Randomly flips images horizontally (left↔right). |
| `fill_mode='nearest'` | Nearest pixel | When rotating/zooming creates empty pixels, fill with nearest existing pixel value. |

### Training Data Generator:

```python
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=42
)
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `dataset_path` | Folder path | Root folder containing class subfolders |
| `target_size=(256, 256)` | 256×256 pixels | **Resize all images** to same dimensions (required for neural networks) |
| `batch_size=32` | 32 images | **Mini-batch:** Process 32 images at once before updating weights |
| `class_mode='categorical'` | One-hot | Labels as vectors: `[0,0,1,0,...]` instead of single number |
| `subset='training'` | 80% | Use the training portion (defined by validation_split) |
| `seed=42` | Random seed | **Reproducibility:** Same random split every time |

### Why batch_size=32?

| Batch Size | Pros | Cons |
|------------|------|------|
| **Small (8-16)** | Less memory, more updates | Noisy gradients, slower |
| **Medium (32-64)** | Good balance | Standard choice |
| **Large (128+)** | Stable gradients, faster | Needs more memory, may overfit |

---

### Validation Data Generator:

```python
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)
```

**Same as training, except:**
- `subset='validation'` → Uses the remaining 20%
- **No augmentation applied** during validation (uses rescale only)

---

## Cell 4: Preview Batch

```python
# Preview batch
x_batch, y_batch = next(train_data)
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.show()
```

### Line-by-line:

| Code | Explanation |
|------|-------------|
| `next(train_data)` | Get one batch (32 images + labels) from generator |
| `x_batch` | Images array, shape: `(32, 256, 256, 3)` |
| `y_batch` | Labels array, shape: `(32, 15)` - one-hot encoded |
| `plt.figure(figsize=(10,10))` | Create 10×10 inch figure |
| `for i in range(9)` | Show first 9 images |
| `plt.subplot(3,3,i+1)` | Create 3×3 grid, position i+1 |
| `plt.imshow(x_batch[i])` | Display image i |
| `plt.axis('off')` | Hide axis labels |

---

## Cell 5: Model Building & Training (THE CORE)

### Part A: Class Weights

```python
# Compute balanced class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
for k in class_weights:
    class_weights[k] = min(class_weights[k], 5.0)  # cap extreme weights
```

| Code | Explanation |
|------|-------------|
| `train_data.classes` | Array of class indices for all training images: `[0, 0, 1, 2, ...]` |
| `np.unique(...)` | Get unique class indices: `[0, 1, 2, ..., 14]` |
| `class_weight='balanced'` | Calculate weights inversely proportional to class frequency |
| `dict(enumerate(...))` | Convert array to dict: `{0: 1.2, 1: 0.8, 2: 1.5, ...}` |
| `min(..., 5.0)` | Cap maximum weight at 5.0 to prevent instability |

---

### Part B: Base Model (MobileNetV2)

```python
# Build model: full MobileNetV2 fine-tuning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = True  # 🔓 Unfreeze all layers
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `weights='imagenet'` | Pre-trained | Load weights trained on ImageNet (1.4M images, 1000 classes) |
| `include_top=False` | No classifier | Remove the original 1000-class output layer (we'll add our own) |
| `input_shape=(256, 256, 3)` | Image dimensions | Height=256, Width=256, Channels=3 (RGB) |
| `trainable = True` | Fine-tune | Allow all layers to be updated during training |

**What does MobileNetV2 output?**
- Input: `(batch, 256, 256, 3)` - RGB images
- Output: `(batch, 8, 8, 1280)` - Feature maps
- 8×8 spatial resolution, 1280 different features per location

---

### Part C: Custom Classifier Layers

```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])
```

**Layer-by-layer breakdown:**

| Layer | Input Shape | Output Shape | Explanation |
|-------|-------------|--------------|-------------|
| `base_model` | (batch, 256, 256, 3) | (batch, 8, 8, 1280) | Extract features |
| `GlobalAveragePooling2D()` | (batch, 8, 8, 1280) | (batch, 1280) | Average each 8×8 feature map → single value |
| `Dropout(0.4)` | (batch, 1280) | (batch, 1280) | Randomly zero 40% of values during training |
| `Dense(256, activation='relu')` | (batch, 1280) | (batch, 256) | Fully connected: 1280→256 neurons with ReLU |
| `Dropout(0.3)` | (batch, 256) | (batch, 256) | Randomly zero 30% of values |
| `Dense(15, activation='softmax')` | (batch, 256) | (batch, 15) | Output: 15 class probabilities |

**GlobalAveragePooling2D visualization:**
```
Feature Map (8×8):          After GAP:
┌─────────────────┐         
│ 0.2 0.4 0.1 ... │         
│ 0.3 0.5 0.2 ... │   →     0.35 (average of all 64 values)
│ ... ... ... ... │         
└─────────────────┘         
```

**Why GAP instead of Flatten?**
- Flatten: 81,920 → Dense(256) = 21 million parameters!
- GAP: 1,280 → Dense(256) = 328,000 parameters
- Fewer parameters = less overfitting

---

### Part D: Optimizer & Compilation

```python
# Compile with AdamW + label smoothing
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
```

**AdamW Optimizer:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `learning_rate=5e-5` | 0.00005 | Very small - we're fine-tuning, not training from scratch |
| `weight_decay=1e-5` | 0.00001 | L2 regularization - penalizes large weights |

---

### Part E: Callbacks

```python
# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
checkpoint = ModelCheckpoint("best_agrovision_model.keras", monitor="val_accuracy", save_best_only=True)
```

**EarlyStopping:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `monitor='val_accuracy'` | Watch validation accuracy | Stop if this metric stops improving |
| `patience=15` | 15 epochs | Wait 15 epochs before stopping |
| `restore_best_weights=True` | Yes | After stopping, restore weights from best epoch |

**ReduceLROnPlateau:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `monitor='val_loss'` | Watch validation loss | Reduce LR if this stops decreasing |
| `factor=0.2` | Multiply by 0.2 | New LR = old LR × 0.2 |
| `patience=5` | 5 epochs | Wait 5 epochs before reducing |

**ModelCheckpoint:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `"best_agrovision_model.keras"` | Filename | Save model to this file |
| `monitor="val_accuracy"` | Watch val_accuracy | Save when this improves |
| `save_best_only=True` | Only best | Don't save every epoch, only improvements |

---

### Part F: Training

```python
# Train (single stage)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `train_data` | Generator | Training data source |
| `validation_data=val_data` | Generator | Validation data source |
| `epochs=15` | 15 | Maximum training rounds |
| `callbacks=[...]` | 3 callbacks | Control training process |
| `class_weight=class_weights` | Dict | Weight samples by class |

**What `model.fit()` returns:**
```python
history.history = {
    'accuracy': [0.65, 0.85, 0.90, ...],      # Training accuracy per epoch
    'val_accuracy': [0.45, 0.75, 0.88, ...],  # Validation accuracy per epoch
    'loss': [1.5, 0.9, 0.7, ...],              # Training loss per epoch
    'val_loss': [2.0, 1.2, 0.8, ...]           # Validation loss per epoch
}
```

---

## Cell 6: Save Model & Artifacts

```python
# Save best model weights
model.save("best_agrovision_model.keras")

# Save class label mapping
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
```

| Code | Explanation |
|------|-------------|
| `model.save(...)` | Save entire model (architecture + weights + optimizer) |
| `.keras` format | TensorFlow 2.x native format (recommended) |
| `train_data.class_indices` | Dict: `{'Pepper__bell___Bacterial_spot': 0, ...}` |
| `json.dump(...)` | Save dict as JSON file |
| `pickle.dump(...)` | Save Python object as binary file |

---

## Cell 7: Evaluate Model

```python
# Evaluate best model
best_model = load_model("best_agrovision_model.keras")
val_loss, val_acc = best_model.evaluate(val_data)
print(f"Validation Accuracy (Best Model): {val_acc*100:.2f}%")
```

| Code | Explanation |
|------|-------------|
| `load_model(...)` | Load saved model from disk |
| `.evaluate(val_data)` | Run validation data through model, calculate loss & accuracy |
| Returns tuple | `(loss_value, accuracy_value)` |
| `val_acc*100:.2f` | Convert 0.99 → 99.00% with 2 decimal places |

---

## Cell 8: Confusion Matrix & Classification Report

```python
# 2. Create a CLEAN validation generator (No augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # <--- THIS IS THE KEY FIX
)
```

**Why `shuffle=False`?**
- For confusion matrix, we need predictions aligned with true labels
- If shuffled, `y_pred[0]` might not correspond to `y_true[0]`
- `shuffle=False` ensures images are processed in consistent order

```python
# 4. Predict
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes
```

| Variable | Shape | Explanation |
|----------|-------|-------------|
| `y_pred` | (N, 15) | Probabilities for each class |
| `y_pred_classes` | (N,) | Predicted class index (0-14) |
| `y_true` | (N,) | True class index (0-14) |
| `np.argmax(..., axis=1)` | Along columns | Get index of maximum value in each row |

---

### Confusion Matrix:

```python
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys())
```

| Code | Explanation |
|------|-------------|
| `confusion_matrix(y_true, y_pred)` | Creates 15×15 matrix |
| `cm[i][j]` | Count of samples with true class i, predicted as class j |
| `annot=True` | Show numbers in cells |
| `fmt='d'` | Format as integers |
| `cmap='Blues'` | Color scheme (higher = darker blue) |

**Reading the matrix:**
- Diagonal = correct predictions (should be high)
- Off-diagonal = errors (should be low)

---

### Classification Report:

```python
print(classification_report(y_true, y_pred_classes, target_names=val_data.class_indices.keys()))
```

**Metrics explained:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all predicted positive, how many are actually positive? |
| **Recall** | TP / (TP + FN) | Of all actual positive, how many did we predict correctly? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **Support** | Count | Number of samples in this class |

---

# 🌐 SECTION 2: app.py (Streamlit Web App)

---

## Imports

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import plotly.express as px
import pandas as pd
from datetime import datetime
```

| Import | Purpose |
|--------|---------|
| `streamlit` | Web framework for ML apps |
| `tensorflow` | Deep learning |
| `numpy` | Array operations |
| `load_model` | Load our trained model |
| `image` | Image preprocessing |
| `PIL.Image` | Python Imaging Library |
| `plotly.express` | Interactive charts |
| `pandas` | Data manipulation |
| `datetime` | Timestamps for history |

---

## Page Configuration

```python
st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `page_title` | "AgroVision AI" | Browser tab title |
| `page_icon` | "🧬" | Favicon in browser tab |
| `layout` | "wide" | Use full screen width |
| `initial_sidebar_state` | "collapsed" | Sidebar hidden by default |

---

## Session State

```python
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None
```

**What is Session State?**
- Streamlit reruns entire script on every interaction
- Session state persists data between reruns
- Without it, history would reset on every click

---

## Model Loading with Caching

```python
@st.cache_resource
def load_ai_model():
    return load_model("best_agrovision_model.keras")

@st.cache_data
def load_class_indices():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}
```

**Caching Decorators:**

| Decorator | Purpose | Persists |
|-----------|---------|----------|
| `@st.cache_resource` | Cache ML models, DB connections | Across all users |
| `@st.cache_data` | Cache data (dicts, DataFrames) | Per user |

**Why cache?**
- Without cache: Load 50MB model on EVERY interaction (slow!)
- With cache: Load once, reuse forever (fast!)

---

## Prediction Logic (THE CORE)

```python
with st.spinner("Processing bio-metric data..."):
    # Prediction Logic
    img = image_pil.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_values = predictions[top_3_indices]
    
    predicted_index = top_3_indices[0]
    confidence = top_3_values[0]
    predicted_class = index_to_class[predicted_index]
```

**Step-by-step transformation:**

| Step | Code | Shape | Explanation |
|------|------|-------|-------------|
| 1 | `image_pil` | (H, W, 3) | Original uploaded image (any size) |
| 2 | `.resize((256, 256))` | (256, 256, 3) | Resize to model's expected input |
| 3 | `img_to_array(img)` | (256, 256, 3) | Convert PIL → NumPy array |
| 4 | `np.expand_dims(..., axis=0)` | (1, 256, 256, 3) | Add batch dimension (model expects batches) |
| 5 | `/= 255.0` | (1, 256, 256, 3) | Normalize to 0-1 range |
| 6 | `model.predict(...)` | (1, 15) | Get probabilities for 15 classes |
| 7 | `[0]` | (15,) | Extract first (only) prediction |

**Getting top 3 predictions:**
```python
predictions = [0.02, 0.01, 0.85, 0.05, 0.03, 0.01, ...]  # 15 probabilities

# argsort() returns indices that would sort the array
predictions.argsort() = [1, 5, 0, 4, 3, ...]  # Indices sorted by value (ascending)

# [-3:] gets last 3 (highest values)
# [::-1] reverses to descending order
top_3_indices = [2, 3, 4]  # Indices of top 3 classes
```

---

## Display Results

```python
if confidence > 0.6:
    st.success(f"### Detected: {display_name}")
    
    m1, m2 = st.columns(2)
    m1.metric("Confidence Score", f"{confidence*100:.1f}%", delta="High Accuracy")
    m2.metric("Processing Time", "0.4s", delta="Real-time")
    
    # Draw Chart
    fig = px.bar(chart_data, x="Probability", y="Condition", orientation='h', ...)
    st.plotly_chart(fig, use_container_width=True)
    
    # Solution
    solution = disease_solutions.get(predicted_class, "No specific solution found.")
    st.info(solution)
```

| Code | Effect |
|------|--------|
| `st.success(...)` | Green success box |
| `st.metric(...)` | Large number with label and delta |
| `px.bar(...)` | Plotly horizontal bar chart |
| `st.plotly_chart(...)` | Embed interactive chart |
| `.get(key, default)` | Safe dict access with fallback |

---

## Save to History

```python
if st.session_state['last_file'] != uploaded_file.name:
    current_result = {
        "name": display_name,
        "confidence": f"{confidence*100:.1f}%",
        "solution": solution,
        "time": datetime.now().strftime("%H:%M:%S"),
        "filename": uploaded_file.name,
    }
    
    st.session_state['history'].insert(0, current_result)
    st.session_state['last_file'] = uploaded_file.name
```

**Why check `last_file`?**
- Streamlit reruns script on every interaction
- Without check, same file would be added multiple times

---

# 🔑 CRITICAL CONCEPTS SUMMARY

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Transfer Learning** | Using pre-trained MobileNetV2 | Don't need millions of images |
| **Fine-tuning** | Unfreezing base model layers | Adapts ImageNet features to plants |
| **Data Augmentation** | Artificial image variations | Prevents overfitting |
| **Class Weights** | Balance imbalanced classes | Fair learning for all classes |
| **Label Smoothing** | Soft labels instead of hard | Prevents overconfidence |
| **Dropout** | Random neuron deactivation | Regularization technique |
| **EarlyStopping** | Stop when not improving | Prevents overfitting |
| **Batch Processing** | Process 32 images at once | Memory efficiency |
| **Softmax** | Probability distribution | Interpretable outputs |
| **Confusion Matrix** | Error visualization | Understand model mistakes |

---

## 📝 PROJECT SUMMARY

| Component | Technology | Purpose |
|-----------|------------|---------|
| Base Model | MobileNetV2 | Feature extraction |
| Training | TensorFlow/Keras | Deep learning framework |
| Web App | Streamlit | User interface |
| Data | PlantVillage Dataset | 15 classes of plant diseases |
| Accuracy | ~99% | Validation performance |
| Output | Disease + Treatment | Actionable recommendations |

Your project demonstrates:
- ✅ Transfer Learning
- ✅ Data Augmentation
- ✅ Class Imbalance Handling
- ✅ Model Evaluation (Confusion Matrix, Classification Report)
- ✅ Web Deployment (Streamlit Cloud)
- ✅ Real-world Application

---

*Document generated for AgroVision Project - Plant Disease Detection System*
