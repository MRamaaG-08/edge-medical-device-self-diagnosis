"""Step 11: Generate confusion matrices for trained classifiers."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os

os.makedirs('figures', exist_ok=True)

df          = pd.read_csv('data/feature_dataset.csv')
features    = [c for c in df.columns if c != 'label']
X           = df[features].values
y           = df['label'].values
class_names = ['CLEAN','NOISY','DRIFTED','ATTENUATED','SPIKED']

with open('models/scaler.pkl','rb') as f: scaler = pickle.load(f)
with open('models/random_forest.pkl','rb') as f: rf = pickle.load(f)
with open('models/svm.pkl','rb') as f: svm = pickle.load(f)

from sklearn.model_selection import StratifiedShuffleSplit
sss1 = StratifiedShuffleSplit(n_splits=1,test_size=0.30,random_state=42)
_,temp_idx = next(sss1.split(X,y))
sss2 = StratifiedShuffleSplit(n_splits=1,test_size=0.50,random_state=42)
_,test_idx = next(sss2.split(X[temp_idx],y[temp_idx]))
X_test = scaler.transform(X[temp_idx][test_idx])
y_test = y[temp_idx][test_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (name, model) in zip(axes, [('Random Forest', rf), ('SVM', svm)]):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', ax=ax, linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {name}',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=150)
plt.show()
print("Saved figures/confusion_matrices.png")
