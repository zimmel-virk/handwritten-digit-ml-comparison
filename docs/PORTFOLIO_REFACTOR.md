# Portfolio refactor provenance

This repository was reconstructed from the supplied **CM3015 Project – Handwritten Digits** PDF/code appendix.

Preserved:
- scikit-learn Digits dataset and train/test split
- train-only StandardScaler usage
- NumPy kNN implementation
- NumPy multiclass linear-regression training approach
- NumPy degree-2 polynomial feature expansion
- 20-component PCA configuration
- Decision Tree baseline
- reported hyperparameters and final printed metrics

Changed:
- notebook cells were reorganized into reusable modules
- duplicate helper definitions were consolidated
- generated results were standardized under `results/generated/`

No new model family was added.
