# Benchmark for Anomaly Detection on Spectral Data Streams - BV4

This project is an updated benchmark (BV4) for anomaly detection on spectral data streams from Optical Emission Spectrometry. We evaluate and compare recent multivariate tabular methods against the Online Bootstrapping K-Nearest Neighbor algorithm.


## Installation

- Step 1: System-Wide Prerequisites
Before installing the Python packages, please ensure you have the following system-level tools installed:

    - Python 3.11.2

    - C++ Build Tools: Required to compile dependencies in dSalmon.

        - On Ubuntu: sudo apt-get install build-essential

        - On Windows: Install "C++ build tools".

    - Java (JDK): Required to run the capymoa package.

- Step 2: Evaluation Environment (env_spectra)


This environment is for running the different experiments.

```bash
# Create the environment
python3 -m venv env_spectra

# Activate the environment
source env_spectra/bin/activate
```

You can install all the necessary packages using pip:

```bash
pip install -r requirements.txt
```

- Step 3: Analysis Environment (env_analysis) This separate environment is only for analysing result-generation scripts.

```bash
# Create the environment
python3 -m venv env_analysis

# Activate the new environment
source env_analysis/bin/activate

# Install vus autorank capymoa and openpyxl
pip install vus==0.0.6 autorank==1.3.0 capymoa==0.9.0 openpyxl==3.1.5 

pip install -r requirements.txt
```

