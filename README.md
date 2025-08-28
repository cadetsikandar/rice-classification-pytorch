# 🌾 Rice Grain Classification using PyTorch

## 📌 Overview
This project implements a **binary classification system** for rice grains (Jasmine vs. Gonen) using **PyTorch**.  
It demonstrates preprocessing of tabular features, building a deep learning model, training, and evaluating performance.

---

## 🎯 Problem Statement
Manual classification of rice varieties is time-consuming and error-prone.  
This project aims to **automate rice classification** using numerical features derived from grain images, achieving higher accuracy and efficiency.

---

## 🧠 Dataset
- Source: [Kaggle Rice Classification Dataset](https://www.kaggle.com/datasets/muratkokludataset/riceclassification)  
- Size: ~18,000 samples  
- Features:  
  - `Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`,  
  - `EquivDiameter`, `Extent`, `Perimeter`, `Roundness`, `AspectRatio`  
- Target: `Class` (0 = Gonen, 1 = Jasmine)

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn  
- **Tools**: Jupyter Notebook / Colab  

---

## 🔍 Approach
1. **Data Preprocessing**  
   - Load CSV dataset  
   - Normalize features  
   - Train-test split  

2. **Model Architecture**  
   - Feed-forward neural network (PyTorch)  
   - Hidden layers with ReLU activations  
   - Output: Binary classification (sigmoid)  

3. **Training**  
   - Optimizer: Adam  
   - Loss: Binary Cross-Entropy  
   - Metrics: Accuracy, Precision, Recall, F1-score  

4. **Evaluation**  
   - Accuracy on test set  
   - Confusion matrix  
   - ROC Curve  

---

## 📈 Results
- Achieved **98% accuracy** on test set  
- Model outperformed baseline ML classifiers (Logistic Regression, Random Forest)  
- Training and evaluation curves are available in `results/`  

---

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/your-username/rice-classification-pytorch.git
cd rice-classification-pytorch

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/Rice_classification.ipynb
