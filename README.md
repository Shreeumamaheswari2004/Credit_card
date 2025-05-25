**Credit Card Fraud Detection System using Mahalanobis Distance, Hybrid Sampling, and Random Forest Algorithm**

This project presents an advanced Credit Card Fraud Detection System that leverages powerful machine learning techniques, robust data preprocessing methods, and a user-friendly web interface. The system is designed to detect fraudulent credit card transactions with high accuracy and efficiency by combining statistical analysis and ensemble learning.


---

**Key Features:**

Hybrid Sampling with SMOTE-ENN: Effectively handles class imbalance by synthesizing new minority class samples and removing noisy data.

Mahalanobis Distance: Enhances outlier detection, improving fraud identification accuracy.

Random Forest and KNN Ensemble Model: Combines the strengths of Random Forest (robustness and feature importance) and KNN (local pattern recognition) for superior prediction.

Principal Component Analysis (PCA): Reduces data dimensionality to optimize performance and speed.

Flask + Streamlit Web Application: Offers a simple interface for users to upload data, view predictions, and evaluate model metrics.

Comprehensive Evaluation: Performance assessed using Accuracy, Precision, Recall, F1-score, ROC Curve, and Confusion Matrix.



---

**System Workflow:**

1. Data Preprocessing: Cleans and encodes transaction data; handles missing values.


2. Feature Reduction: PCA is applied to retain the most relevant information while simplifying the model.


3. Data Balancing: SMOTE-ENN balances the dataset to address fraud rarity.


4. Model Training: Random Forest, KNN, and a hybrid RF + KNN model are trained on the data.


5. Prediction and Evaluation: The system predicts fraud and evaluates results using multiple metrics.


6. Web Integration: A user-facing app allows real-time interaction and insight retrieval.




---

**Technologies Used:**

Language: Python

Libraries: pandas, sklearn, pyplot, pyqrcode, numpy

Frameworks: Flask, Streamlit

IDE: Anaconda (Spyder)



---

**Use Cases:**

Real-time fraud detection for banks and payment processors

Historical transaction analysis for audits

Educational tool for understanding fraud detection in machine learning



---

**Future Enhancements:**

Real-time detection engine using streaming data (e.g., Kafka)

Integration of deep learning models such as LSTMs and Autoencoders

Explainable AI (XAI) to improve model interpretability

Deployment via cloud services for scalability
