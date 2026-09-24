# Federated Learning for Multimodal Human Activity Recognition

Research code exploring multimodal human activity recognition with centralized and federated learning approaches.

The project experiments with multiple sensing modalities, including LiDAR, infrared, mmWave, RGB, Wi-Fi CSI, and depth data. It includes modality-specific encoders, classification heads, feature/decision fusion, evaluation utilities, and federated-learning experiments.

## Repository structure

- `data/` — preprocessing and data augmentation utilities
- `mmfi_lib/` — dataset loading helpers
- `models/` — encoders, classifiers, and fusion layers
- `experiments/` — training and evaluation functions
- `scripts/` — centralized and federated experiment entry points
- `utils/` — shared argument/configuration helpers
- `run_overnight.py` — repeated experiment runner

## Data

The training scripts expect preprocessed modality data derived from the MM-Fi multimodal human activity recognition dataset. Large source/preprocessed datasets and model checkpoints are not included in this repository.

Several scripts expect modality-specific pickle files such as `mmwave.pkl` and `wifi-csi.pkl` in the working directory.

## Dependencies

The code is Python-based and uses PyTorch. Other observed dependencies include NumPy, scikit-learn, torchmetrics, and torchsummary.

Hardware acceleration is selected automatically where supported by the individual scripts.

## Running an experiment

Run scripts from the repository root so the local packages can be imported. For example:

```bash
PYTHONPATH=. python scripts/centralized_multimodal_decision.py
```

`run_overnight.py` provides a convenience runner for repeating the multimodal decision-fusion experiment and appending output to `multimodal.txt`.

## Status

This repository is a research project and experiment workspace rather than a packaged library. Results depend on the associated dataset, preprocessing, experiment configuration, and available hardware.
