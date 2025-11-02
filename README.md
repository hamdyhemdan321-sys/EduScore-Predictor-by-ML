# EduScore Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-green.svg)]([https://github.com/yourusername/eduscore-predictor](https://github.com/hamdyhemdan321-sys/EduScore-Predictor-by-ML))

## Project Overview

EduScore Predictor is a machine learning project aimed at classifying students' performance in mathematics as "Pass" (score ≥ 60) or "Fail" (score < 60) based on various demographic and academic features. The project uses traditional ML models—Logistic Regression and Random Forest—to predict outcomes and evaluates their performance using key metrics like accuracy, precision, recall, F1-score, and ROC AUC.

This project demonstrates a complete ML workflow: data loading, cleaning, preprocessing, model training, evaluation, and visualization. It's built using Python and scikit-learn, making it a great example for educational purposes or portfolio showcasing.

### Key Features
- **Dataset**: Students' Performance dataset (1000 rows, 8 columns) from Kaggle.
- **Models**: Logistic Regression and Random Forest Classifier.
- **Evaluation**: Comprehensive metrics and visualizations (Confusion Matrix, ROC Curve).
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn.

## Dataset
- **Source**: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (CSV file with 1000 entries).
- **Features Used**:
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch Type
  - Test Preparation Course
  - Reading Score
  - Writing Score
- **Target**: Binary classification of Math Pass (1 = Pass, 0 = Fail) based on Math Score ≥ 60.

## Methodology
1. **Data Cleaning**: Checked for missing values, duplicates, and outliers (none found).
2. **Preprocessing**: One-hot encoding for categorical features, standard scaling for numerical features, and 80/20 train-test split.
3. **Model Training**: Built pipelines for Logistic Regression and Random Forest.
4. **Evaluation**: Computed accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.
5. **Visualizations**: Heatmaps for confusion matrices and plotted ROC curves for model comparison.

## Results
- **Logistic Regression**:
  - Accuracy: 0.88
  - Precision: 0.88
  - Recall: 0.95
  - F1-Score: 0.91
  - ROC AUC: 0.97
- **Random Forest**:
  - Accuracy: 0.86
  - Precision: 0.86
  - Recall: 0.93
  - F1-Score: 0.90
  - ROC AUC: 0.95

Logistic Regression slightly outperformed Random Forest, particularly in ROC AUC and overall accuracy.

For detailed visualizations (Confusion Matrices and ROC Curve), refer to the Jupyter Notebook.

## Installation and Requirements
To run this project locally or on Google Colab:

### Prerequisites
- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

### Setup
1. Clone the repository:
   ```
   git clone [https://github.com/yourusername/eduscore-predictor.git](https://github.com/hamdyhemdan321-sys/EduScore-Predictor-by-ML)
   cd eduscore-predictor
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset: Place `StudentsPerformance.csv` in the root directory (or upload to `/content/` in Colab).

## Usage
1. Open the Jupyter Notebook: `modified_notebook.ipynb`.
2. Run all cells sequentially.
   - The notebook will load the data, perform cleaning/preprocessing, train models, evaluate them, and generate visualizations.
3. Update the Project Report section in the notebook with any custom changes if needed.

Example output from evaluation:
- Metrics printed in the console.
- Visuals displayed inline.

## Future Improvements
- Hyperparameter tuning using GridSearchCV.
- Experiment with advanced models like XGBoost or Neural Networks.
- Feature engineering (e.g., combining scores or adding interactions).
- Deployment as a web app using Streamlit or Flask.

## Contributing
Contributions are welcome! Feel free to fork the repo, make improvements, and submit a pull request. For major changes, please open an issue first to discuss.

## Acknowledgments
- Dataset provided by [Kaggle user spscientist](https://www.kaggle.com/spscientist).
- Built with inspiration from standard ML workflows in education analytics.

If you find this project useful, give it a ⭐ on GitHub! For questions, contact [hamdyhemdan321@gmail.com].
