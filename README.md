
## Titanic Survival Prediction

### Overview
This Python script utilizes machine learning algorithms to predict the survival outcome of passengers aboard the Titanic. It employs three different classifiers: Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression. Additionally, the script calculates conditional probabilities based on various passenger attributes such as gender, age, and passenger class.

### Data Source
The dataset used in this script is sourced from the Titanic Dataset, containing information about passengers including their age, gender, ticket class, embarkation port, etc.

### Preprocessing
1. Feature Engineering: Irrelevant features such as 'Name' and 'Cabin' are removed from the dataset.
2. Label Encoding: Categorical features ('Sex', 'Embarked', 'Ticket') are encoded numerically using LabelEncoder.
3. Train-Test Split: The dataset is split into training and testing sets with a ratio of 70:30 for model evaluation.

### Models Implemented
- Naive Bayes (NB): Gaussian Naive Bayes classifier is trained and evaluated.
- K-Nearest Neighbors (KNN): KNN classifier with k=11 is trained and evaluated.
- Logistic Regression (LR): Logistic Regression classifier with Newton-CG solver is trained and evaluated.

### Evaluation
- Confusion Matrix: Confusion matrices are generated to visualize the performance of each classifier.
- Classification Report: Detailed classification reports are provided for each classifier, including precision, recall, F1-score, and support metrics.
- Elbow Method: The Elbow Method is employed to determine the optimal value of k for the KNN classifier.

### Prediction on New Data
The script includes the prediction of survival outcomes for a new passenger, "Jack Dawson", based on the trained models.

### Conditional Probability Calculation
Conditional probabilities are calculated for various scenarios including gender, age groups, and passenger class to analyze survival likelihood under different conditions.

### Dependencies
The script requires the following Python libraries:
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

### Usage
1. Ensure all dependencies are installed.
2. Run the script in a Python environment.
3. Follow the prompts to input data or observe the analysis results.

### Limitations
- The script assumes the dataset is accurate and complete, which may not always be the case in real-world scenarios.
- It provides insights based on the provided dataset and may not generalize well to other datasets or scenarios without proper validation.

### Future Improvements
- Implementation of cross-validation techniques for more robust model evaluation.
- Exploration of additional machine learning algorithms to potentially improve predictive performance.
- Enhancement of conditional probability analysis to include more diverse scenarios and features.
