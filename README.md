# SBI-Fraud-Detection
# Brief Description
The problem statement (PS) centres on developing robust, interpretable machine learning and data analysis solutions using a complex, noisy dataset that merges anonymised financial transactions and mobile tower connection logs. The challenge has two main tasks:

- **Fraud Detection:** Participants must analyze device usage patterns, profile information, and transactional data to identify fraudulent accounts or transactions. The key challenges include handling adversarial behavior (where fraudsters try to mimic normal activity), addressing significant class imbalance (very few fraud cases), and ensuring that the model’s predictions are explainable.

- **Defaulter Localization:** The goal is to use mobile tower connection logs to estimate the last known location of labeled defaulters. Despite the data being noisy, incomplete, and anonymized, solutions should provide traceable and interpretable predictions about the most recent digital footprint of defaulters, maintaining temporal consistency and clear inference logic.

Overall, the problem statement requires handling messy, real-world data with missing values and anomalies, demonstrating the ability to extract actionable insights under practical constraints, and delivering solutions that are both resilient to noise and adversarial tactics, and interpretable for real-world application.

---

# Project Setup

This guide explains how to set up your Python environment for this project.

---

## 1. Create a Virtual Environment

- **Windows:**
python -m venv .venv

- **macOS/Linux:**
python3 -m venv .venv

---

## 2. Activate the Virtual Environment

- **Windows:**
.venv\Scripts\activate

- **macOS/Linux:**
source .venv/bin/activate

---

## 3. Install Dependencies

Make sure you are in the project directory and your virtual environment is activated.

pip install -r requirements.txt

---

## 5. Deactivate the Virtual Environment

When you are done working, deactivate with:

deactivate

---

# Frameworks Used
- TensorFlow
- Python
- Keras
- Pandas
- NumPy
- matplotlib
