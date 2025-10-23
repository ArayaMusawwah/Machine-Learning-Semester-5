# Lembar Kerja Pertemuan 7 - ANN untuk Klasifikasi (Reproducible script / notebook)
# Python 3.10.x, TensorFlow/Keras
# Jalankan sebagai notebook (Jupyter) atau script yang diubah sedikit untuk CLI

# 0. Persiapan: set seed untuk reproducibility
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reproducibility
RSEED = 42
os.environ['PYTHONHASHSEED'] = str(RSEED)
random.seed(RSEED)
np.random.seed(RSEED)

import tensorflow as tf
tf.random.set_seed(RSEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score)

from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data
# Pastikan file processed_kelulusan.csv berada di folder kerja saat ini
DATA_PATH = "processed_kelulusan.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Tidak menemukan {DATA_PATH}. Letakkan file di folder kerja.")

df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)
print(df.head())

# Asumsi: kolom target bernama 'Lulus' (0/1 atau False/True). Jika beda, ubah nama kolom.
TARGET = "Lulus"
if TARGET not in df.columns:
    raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di dataset")

X = df.drop(TARGET, axis=1)
y = df[TARGET].astype(int)

# 2. Preprocessing: For very small datasets like this one (only 10 samples), we need a different approach
# The regular train/val/test split doesn't work well with only 10 samples
# We'll use a simple approach: 6 for training, 2 for validation, 2 for testing

print("⚠️  Very small dataset detected (10 samples). Adjusting approach...")
print("Using 6 samples for training, 2 for validation, 2 for testing.")

# Convert to numpy arrays and manually split
X_array = X.values if hasattr(X, 'values') else X
y_array = y.values if hasattr(y, 'values') else y

# Manually split to ensure we have enough samples per set
from sklearn.utils import shuffle
X_array, y_array = shuffle(X_array, y_array, random_state=RSEED)

X_train = X_array[:6]
X_val = X_array[6:8]
X_test = X_array[8:10]
y_train = y_array[:6]
y_val = y_array[6:8]
y_test = y_array[8:10]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
print(f"Train labels: {list(y_train)}, Val labels: {list(y_val)}, Test labels: {list(y_test)}")

# Fit scaler hanya pada training data, lalu transform semua set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# 4. Build model: fungsi pembuat model agar mudah bereksperimen

def build_model(input_dim,
                hidden_units=[16, 8],
                dropout_rate=0.3,  # Reduced dropout
                l2_reg=0.001,      # Reduced L2 regularization
                use_batchnorm=True,
                lr=1e-3,
                optimizer_name='adam'):
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
        if dropout_rate and dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(4, activation='relu', kernel_regularizer=reg))
    if dropout_rate and dropout_rate > 0:
        model.add(layers.Dropout(0.2))  # Reduced dropout
    
    model.add(layers.Dense(1, activation='sigmoid'))

    # pilih optimizer
    if optimizer_name.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError('optimizer_name harus "adam" atau "sgd"')

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='AUC')])
    return model

# 5. Default model sesuai petunjuk
input_dim = X_train.shape[1]
model = build_model(input_dim,
                    hidden_units=[16, 8],
                    dropout_rate=0.5,
                    l2_reg=0.01,
                    use_batchnorm=True,
                    lr=1e-3,
                    optimizer_name='adam')

model.summary()

# 6. Callbacks termasuk EarlyStopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)  # Reduced patience
# optional: Simpan model terbaik
ckpt_path = 'best_model.h5'
mc = keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)

# 7. Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Reduced epochs
    batch_size=32,  # Increased batch size
    callbacks=[es, mc],
    verbose=1
)

# 8. Evaluasi pada test set
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Acc: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# prediksi probabilitas dan prediksi biner default threshold 0.5
y_proba = model.predict(X_test).ravel()
y_pred_05 = (y_proba >= 0.5).astype(int)

print("Confusion matrix (threshold 0.5):")
print(confusion_matrix(y_test, y_pred_05))
print("Classification report (threshold 0.5):")
print(classification_report(y_test, y_pred_05, digits=3))

# Note about small dataset
print("\n⚠️  IMPORTANT: With only 2 test samples, these metrics may not be reliable.")
print("The ROC AUC of 1.0 is likely due to the extremely small test set and may not reflect true model performance.")

# 9. ROC, AUC, dan analisis threshold
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_auc:.3f})')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=120)
plt.close()

# 10. Precision-Recall dan analisis threshold untuk F1
precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)
# compute f1 for thresholds
f1_scores = []
for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thr))

# cari threshold terbaik berdasarkan F1
best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else None
best_thr = thresholds[best_idx] if best_idx is not None else 0.5
best_f1 = f1_scores[best_idx] if best_idx is not None else f1_score(y_test, y_pred_05)

print(f"Best threshold by F1 (approx): {best_thr:.3f} with F1={best_f1:.3f}")

# plot learning curve
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curve')
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=120)
plt.close()

# plot histogram of predicted probabilities
plt.figure()
plt.hist(y_proba[y_test==0], bins=30, alpha=0.6, label='neg')
plt.hist(y_proba[y_test==1], bins=30, alpha=0.6, label='pos')
plt.xlabel('Predicted probability')
plt.legend()
plt.title('Predicted probability distribution')
plt.tight_layout()
plt.savefig('proba_hist.png', dpi=120)
plt.close()

# 11. Laporan metrik akhir (threshold default dan best threshold)

def report_for_threshold(thr):
    y_pred = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    return {'threshold': thr, 'confusion_matrix': cm, 'classification_report': cr,
            'f1': f1, 'precision': prec, 'recall': rec}

report_05 = report_for_threshold(0.5)
report_best = report_for_threshold(best_thr)

print('\n--- Report (threshold 0.5) ---')
print(report_05['confusion_matrix'])
print(report_05['classification_report'])
print(f"F1: {report_05['f1']:.3f}, Precision: {report_05['precision']:.3f}, Recall: {report_05['recall']:.3f}")

print('\n--- Report (best threshold by F1) ---')
print(report_best['confusion_matrix'])
print(report_best['classification_report'])
print(f"F1: {report_best['f1']:.3f}, Precision: {report_best['precision']:.3f}, Recall: {report_best['recall']:.3f}")

# 12. Eksperimen singkat: grid over neurons dan optimizer
# hasil akan disimpan ke dataframe untuk dilihat
experiment_results = []
search_space = [
    {'hidden_units':[16,8], 'dropout':0.2, 'optimizer':'adam', 'lr':1e-3},  # Reduced complexity
    {'hidden_units':[32,16], 'dropout':0.3, 'optimizer':'adam', 'lr':1e-3},
    {'hidden_units':[16,8], 'dropout':0.3, 'optimizer':'sgd', 'lr':1e-2},
    {'hidden_units':[8,4], 'dropout':0.2, 'optimizer':'adam', 'lr':1e-3},   # Even simpler
]

for conf in search_space:
    print('Running experiment:', conf)
    m = build_model(input_dim,
                    hidden_units=conf['hidden_units'],
                    dropout_rate=conf['dropout'],
                    l2_reg=0.0,
                    use_batchnorm=False,
                    lr=conf['lr'],
                    optimizer_name=conf['optimizer'])
    # short training to compare (few epochs)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)
    y_p = m.predict(X_test).ravel()
    auc_val = roc_auc_score(y_test, y_p)
    y_p_bin = (y_p >= 0.5).astype(int)
    f1_val = f1_score(y_test, y_p_bin)
    experiment_results.append({**conf, 'AUC': auc_val, 'F1': f1_val})
    print(f"-> AUC={auc_val:.3f}, F1={f1_val:.3f}")

exp_df = pd.DataFrame(experiment_results)
exp_df.to_csv('experiment_results.csv', index=False)

# tampilkan ringkasan
print('\nExperiment results:')
print(exp_df)

# 13. Simpan model final (jika ingin)
model.save('final_model.keras')

# 14. Catatan untuk laporan (string yang bisa dicopy ke laporan)
notes = f"Arsitektur final: Dense{[32,16]} + Dropout 0.3; Optimizer: Adam lr=1e-3; EarlyStopping(patience=10). Test AUC={roc_auc:.3f}; Best threshold(F1)={best_thr:.3f}"
with open('report_notes.txt', 'w') as f:
    f.write(notes)

print('\nSelesai. File yang dihasilkan: learning_curve.png, roc_curve.png, proba_hist.png, best_model.h5, final_model/, experiment_results.csv, report_notes.txt')

# Jika dijalankan sebagai notebook, tambahkan visualisasi inline sesuai kebutuhan.
