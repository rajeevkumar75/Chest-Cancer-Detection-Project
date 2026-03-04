# Chest Cancer Detection (Adenocarcinoma) — CNN + Streamlit

Link: https://rajeevkumar75-chest-cancer-detection-projec-templatesapp-br12sc.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

This repository implements an end‑to‑end deep learning pipeline to detect Adenocarcinoma (a type of lung cancer) from chest CT/X‑ray images using transfer learning on VGG16, and provides a Streamlit UI for inference. The codebase is organized as a reusable Python package with clear configuration and logging.

## Overview
- Data ingestion downloads and extracts a CT‑Scan dataset from Google Drive.
- Base model preparation loads VGG16 (Imagenet weights), freezes layers, and adds a small classification head.
- Model training uses Keras ImageDataGenerator with optional augmentation and saves the trained model.
- Streamlit app performs inference on uploaded images and presents probability and status.

## Project Structure
```
Chest cancer detection project/
├─ config/
│  └─ config.yaml                          # paths, dataset URL, artifacts
├─ params.yaml                              # training hyperparameters
├─ src/cnnClassifier/
│  ├─ __init__.py                           # logger setup
│  ├─ components/
│  │  ├─ data_ingestion.py                  # gdown download + unzip
│  │  ├─ prepare_base_model.py              # VGG16 + head, save base/updated
│  │  └─ model_trainer.py                   # generators + training loop
│  ├─ config/configuration.py               # builds typed configs from YAML
│  ├─ constants/__init__.py                 # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│  ├─ entity/config_entity.py               # dataclasses for configs
│  ├─ pipeline/
│  │  ├─ data_ingestion.py                  # stage runner
│  │  ├─ prepare_model.py                   # stage runner
│  │  └─ model_trainer.py                   # stage runner
│  └─ utils/common.py                       # YAML/JSON IO, joblib, helpers
├─ templates/app.py                         # Streamlit inference UI
├─ artifacts/                               # generated models and data (gitignored)
├─ requirements.txt
├─ setup.py                                 # package metadata
├─ main.py                                  # orchestrates 3 pipeline stages
└─ README.md
```

Useful code entry points:
- Pipeline orchestrator: [main.py](file:///z:/Chest%20cancer%20detection%20project/main.py)
- Configuration builder: [configuration.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/config/configuration.py)
- Data ingestion: [data_ingestion.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/components/data_ingestion.py)
- Base model prep: [prepare_base_model.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/components/prepare_base_model.py)
- Training: [model_trainer.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/components/model_trainer.py)
- Streamlit app: [app.py](file:///z:/Chest%20cancer%20detection%20project/templates/app.py)

## Quickstart
- Prerequisites: Python 3.9+ (TensorFlow 2.x), pip; optional GPU for training.
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

- (Optional) install the package locally:
```bash
pip install -e .
```

### Train the Model
The training pipeline has three stages executed by `main.py`:
1) Data Ingestion → downloads the zip from Google Drive and extracts into `artifacts/data_ingestion/unziped_data/`
2) Prepare Base Model → loads VGG16 and saves base/updated models to `artifacts/prepare_base_model/`
3) Model Training → builds train/valid generators and saves the final model to `artifacts/trained_model/model.h5`

Run:
```bash
python main.py
```

Training expects a directory structure under:
```
artifacts/data_ingestion/unziped_data/CT-Scan data/
  ├─ class_1/
  └─ class_2/
```
This is the standard Keras `flow_from_directory` format. The dataset URL is defined in `config/config.yaml`.

### Run the Streamlit App
After training produces `artifacts/trained_model/model.h5`, start the UI:
```bash
streamlit run templates/app.py
```
- Upload a JPG/PNG chest CT/X‑ray.
- The app applies a basic sanity check to avoid non‑medical images.
- Outputs probability, confidence, and a Positive/Negative verdict.

## Configuration
- Paths and dataset URL: [config.yaml](file:///z:/Chest%20cancer%20detection%20project/config/config.yaml)
- Training hyperparameters: [params.yaml](file:///z:/Chest%20cancer%20detection%20project/params.yaml)
- Constants pointing to config locations: [constants](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/constants/__init__.py)

Configuration is loaded via utilities:
- YAML/JSON helpers: [common.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/utils/common.py#L12-L34)
- Directory creation: [common.py](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/utils/common.py#L37-L49)

## Implementation Notes
- Base model: VGG16 with Imagenet weights, `include_top=False`; custom Dense head added.
- Generators: Keras `ImageDataGenerator` with optional augmentation and validation split.
- Loss/metrics: compiled for training with Adam and `binary_crossentropy`, accuracy.
- Logging: all stages log to console and to [logs_run.log](file:///z:/Chest%20cancer%20detection%20project/src/cnnClassifier/__init__.py#L9-L21) in `logs/`.
- Artifacts: models and intermediate files are stored under `artifacts/` (gitignored).

## Troubleshooting
- gdown download: ensure the Google Drive link in `config.yaml` is accessible; network required.
- TensorFlow install: on Windows, use a compatible Python and VC++ runtime; for GPU, install CUDA/cuDNN per TensorFlow docs.
- Missing model: the Streamlit app will warn if `artifacts/trained_model/model.h5` is not found.

## License and Citation
- License: Add a LICENSE file (e.g., MIT/Apache‑2.0) to clarify usage and distribution.
- If you use this project in academic work, please cite the repository URL.

## Acknowledgments
- Built with TensorFlow/Keras and Streamlit.
- Transfer learning approach inspired by standard medical imaging classification practices.
