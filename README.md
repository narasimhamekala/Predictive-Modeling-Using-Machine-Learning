# Predictive Modeling Using Machine Learning

**Objective:** Build a supervised model to predict outcomes from data, train/test it, and visualize performance.

**Dataset:** Breast Cancer Wisconsin (scikit-learn built-in), 569 samples, 30 features. Target: malignant vs benign.

**Models:** Logistic Regression, Decision Tree, Random Forest (80/20 stratified split, 5-fold CV).

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Decision Tree | 0.9386 | 0.9577 | 0.9444 | 0.9510 | 0.9342 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |

Best model: Logistic Regression (about 98% accuracy).

## Files
- `Task2_Predictive_Modeling.ipynb` - full notebook (run top to bottom)
- `predictive_modeling.py` - same code as a script
- `01_class_distribution.png`, `02_correlation_heatmap.png`, `03_confusion_matrices.png`, `04_roc_curves.png`, `05_feature_importance.png`
- `model_comparison.csv` - metrics table

## Run
```
pip install pandas numpy matplotlib seaborn scikit-learn
python predictive_modeling.py
```
