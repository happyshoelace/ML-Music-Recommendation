# ML Music Recommendation (Autoencoder Experiments)

This repository contains an experimental autoencoder-based workflow for music / feature-based recommendation. The core work is in the Jupyter notebook `main.ipynb`. Pretrained Keras model files are in the `models/` folder and a sample dataset is `dataset.csv`.

## What this repo contains

- `main.ipynb` — Primary notebook with data loading, preprocessing, autoencoder/encoder training and evaluation, and visualization code.
- `dataset.csv` — Tabular input data used in the notebook (features per item / track).
- `models/` — Saved Keras model artifacts (autoencoders and encoder sub-models). Filenames like `encoder_model1.keras` and `autoencoder_model3.keras` are included.
- `readme.md` — This file (human readable guide).

## Goals

- Train autoencoders on the feature dataset to learn compressed (latent) representations.
- Use the encoder outputs as embeddings for downstream tasks (clustering, nearest neighbours, recommendations).
- Provide quick utilities for visualizing the learned latent space and computing reconstruction error.

## Quickstart (Windows PowerShell)

1. Create a Python environment and install dependencies (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Start Jupyter and open the notebook:

```powershell
jupyter notebook main.ipynb
```

3. Run the notebook cells in order. The notebook includes cells to load `dataset.csv`, preprocess, train and evaluate models, and to visualize latent embeddings.

## Loading and using saved models

Saved Keras models are in the `models/` directory. Typical usage:

```python
from tensorflow import keras
encoder = keras.models.load_model('models/encoder_model1.keras')
autoencoder = keras.models.load_model('models/autoencoder_model1.keras')

# produce latents and reconstructions
X_sample = df[numerical_trainSet.columns].values[:100]
X_latents = encoder.predict(X_sample)
X_recon = autoencoder.predict(X_sample)
```

Important: the `encoder` model returns the low-dimensional latent vectors. The `autoencoder` (or full model) returns reconstructions that match the original input dimensionality. When computing reconstruction error, compare the original inputs to `autoencoder.predict(...)`, not to the encoder outputs.

## Visualizing latent space

If the encoder was trained to output more than 2 dimensions, plotting `X_latents[:,0]` vs `X_latents[:,1]` may not reveal structure. Use a projection (PCA / t-SNE / UMAP) before plotting:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X2 = pca.fit_transform(X_latents)
plt.scatter(X2[:,0], X2[:,1], s=6, alpha=0.8)
plt.title('PCA(2) of encoder outputs')
plt.show()
```

For non-linear separations, try t-SNE or UMAP on a subsample (they are slower).

## Compute reconstruction error (what to compare)

A common mistake is to compute mean squared error between the original input and the encoder output. Instead compute it between the original input and the autoencoder reconstruction:

```python
from sklearn.metrics import mean_squared_error
X_recon = autoencoder.predict(numerical_trainSet.values)
recon_mse = mean_squared_error(numerical_trainSet.values, X_recon)
print('Reconstruction MSE:', recon_mse)
```

If you see shape-mismatch errors, check the shapes printed by:

```python
print('input shape:', numerical_trainSet.values.shape)
print('recon shape:', X_recon.shape)
print('latents shape:', X_latents.shape)
```

## Common troubleshooting

- Graph looks collapsed / points overlapping:
  - Check for NaNs or Infs in `X_latents`:

    ```python
    import numpy as np
    print(np.isnan(X_latents).sum(), np.isinf(X_latents).sum())
    print(np.nanmin(X_latents, axis=0)[:5], np.nanmax(X_latents, axis=0)[:5])
    ```

  - If many values are constant, verify your preprocessing or model weights.
  - Project a higher-dimensional latent into 2D with PCA/t-SNE/UMAP before plotting.

- Are you using the encoder or autoencoder for predictions?
  - Verify explicitly in the notebook by printing:

    ```python
    print(type(encoder), type(autoencoder))
    print('encoder.predict sample shape:', encoder.predict(sample).shape)
    print('autoencoder.predict sample shape:', autoencoder.predict(sample).shape)
    ```

  - `encoder.predict(...)` should return a smaller array (latents) and `autoencoder.predict(...)` should return an array with the original feature dimension.

- `history.history.clear()` — caution:
  - Calling `history.history.clear()` removes all stored training metrics from the `History` object's in-memory `history` dictionary. This is destructive: you will no longer be able to plot training/validation loss or access per-epoch metrics unless you saved them earlier. If you want to reclaim memory but keep the metrics, make a copy first:

    ```python
    saved_metrics = history.history.copy()
    history.history.clear()
    ```

## Reproducibility

- If you want deterministic behavior, set random seeds for numpy and TensorFlow at the top of the notebook. Note that full determinism with GPU acceleration can be challenging.

---

Notes: when referring to files in this README use the notebook `main.ipynb` for experiments and the `models/` folder for saved Keras models.