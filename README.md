# EBI02201 README

## Project Overview

This repository is dedicated to the "Machine Learning for Drug Discovery" project, focusing on research literature mining to uncover various entities and classes relevant to drug discovery. These include variants, biomarkers, tissues/cell types, adverse events, and assay conditions. The project aims to develop a machine learning pipeline encompassing data collection, preprocessing, model development, evaluation, and optimization for production complexity. For more detailed information, please refer to the project brief.

## Project Organization

```
EBI02201/
├── README.md                - Top-level README for developers.
├── data/
│   ├── external             - Third-party data sources.
│   ├── interim              - Intermediate, transformed data.
│   ├── processed            - Final datasets for modeling.
│   └── raw                  - Original, immutable data.
│
├── models/                  - Serialized models, predictions, summaries.
│
├── analysis/               - Jupyter notebooks and Python analysis files.
│
├── reports/                 - Analysis reports in various formats.
│   └── figures              - Graphics and figures for reports.
│
├── requirements.txt         - Required packages for reproducing the environment.
│
├── setup.py                 - Setup script for installing the project package.
│
└── src/                     - Source code for the project.
    ├── __init__.py          - Makes 'src' a Python module.
    └── ebi02201/            - Importable code files.
```

## Getting Started

### Prerequisites

- **Conda**: For package management and virtual environments. Recommended distribution is [Miniforge](https://github.com/conda-forge/miniforge). [Miniconda](https://conda.io/) and [Anaconda](https://www.anaconda.com/) are also viable but may require licenses for commercial use.

### Setup

1. **Conda Installation**: Install and configure Conda. Ensure `conda activate` works correctly.
2. **Virtual Environment**: Create a Conda environment with `conda env create -f environment.yml --name <env_name>`.
3. **Environment Activation**: Activate the environment with `conda activate <env_name>`.
4. **Jupyter Lab**: For notebook exploration, use `jupyter lab`.

### Data Collection and Preprocessing

Data for this project is gathered using the Europe PMC Annotations API. The dataset focuses on named entity recognition (NER), but you're free to modify the code or use other datasets for classification or Q&A problems.

### Analysis and Model Development

The `analysis` directory contains essential notebooks/scripts. For instance, `00-demo-import-from-source.ipynb` demonstrates how to import functions from the source package. The project involves tasks such as data collection, preprocessing, ML model development, and evaluation.

### Model Evaluation and Complexity Reduction

When evaluating and finalizing models, consider hyperparameter tuning. For production deployment, focus on model compression or knowledge distillation to manage complexity.

### Additional Information

- **Visual Studio Code Setup**: Configure Python environment, enable code checking tools like flake8 and pylint.
- **Data Access**: Instructions on accessing and utilizing data for analysis will be added here.
- **Feedback**: On completion, submit your notebook (including outputs and comments) in a zip format named `<firstname_lastname.zip>` for review.

[//]: # (Links)
[conda]: https://conda.io/
[miniforge]: https://github.com/conda-forge/miniforge
[anaconda-tos]: https://www.anaconda.com/terms-of-service

---

*Note: The detailed machine learning pipeline steps, including environment setup for the FalconFrames project and specific Python libraries usage, are outlined in the provided documentation and should be followed accordingly.*