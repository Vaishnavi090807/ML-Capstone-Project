ML Insight Lab — Machine Learning Capstone
An evaluator-facing Machine Learning capstone project covering classification, regression, and clustering. The project includes Jupyter notebooks for experimentation and a Streamlit dashboard for exploring model results and visualizations.
Important: Re-run the final notebooks from top to bottom before submission and update the results below if the outputs change. The values listed here are the results currently recorded during development.

Project Tracks
1. Classification — IPL First-Innings Score Category
- Goal: Classify an IPL first-innings score as Low, Medium, or High using match/innings features.
- Notebook: notebooks/classification.ipynb
- Dataset: IPL ball-by-ball data stored in data/ (for example, ipl.csv; use the exact filename present in your repository).
- Algorithms: Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Support Vector Classifier, Random Forest, AdaBoost, Gradient Boosting, Bagging, and MLP.
- Evaluation: Accuracy, Precision, Recall, weighted F1-score, confusion matrix, and multiclass ROC-AUC using One-vs-Rest (OvR).
The recorded comparison selected Logistic Regression as the strongest overall model, with accuracy 0.7061, weighted F1-score 0.7044, and ROC-AUC 0.8785. Reconfirm these metrics after the final notebook run.
2. Regression — Bengaluru House Price Prediction
- Goal: Predict house prices from property-related features.
- Notebook: notebooks/regression.ipynb
- Dataset: data/Bengaluru_House_Data.csv
- Algorithms: Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression, and K-Nearest Neighbors Regressor.
- Evaluation: R², RMSE, MAE, and 5-fold cross-validated R² for the two best-performing models.
The development results recorded Lasso Regression with R² 0.5211, RMSE 22.1799, and MAE 16.0332 lakhs. Confirm these results against the final notebook output before submission.
3. Clustering — Market Arrivals
- Goal: Discover groups of similar observations in the market-arrivals dataset without using target labels during model fitting.
- Notebook: Use the final MarketArrivals clustering notebook in notebooks/ (for example, clustering_market_arrivals.ipynb; update this filename if your local file has a different name).
- Dataset: data/MarketArrivals.csv
- Algorithms: K-Means and Agglomerative Hierarchical Clustering, with linkage strategies compared for hierarchical clustering.
- Evaluation: Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index.
- Visualizations: K-Means elbow curve, hierarchical dendrogram, PCA 2D cluster plots, and t-SNE visualization.
