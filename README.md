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
The selected number of K-Means clusters and the final comparison metrics should be taken from the final successful run of the MarketArrivals notebook. Cluster labels are discovered groups, not pre-existing class labels.
Repository Structure
project-root/
├── app/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── Bengaluru_House_Data.csv
│   ├── MarketArrivals.csv
│   └── <IPL dataset CSV>
├── models/
├── notebooks/
│   ├── classification.ipynb
│   ├── regression.ipynb
│   └── <final MarketArrivals clustering notebook>
├── README.md
└── requirements.txt
The filenames above are examples of the intended organization. Keep your actual filenames and update this section if they differ. Avoid committing large, restricted, or sensitive datasets to a public repository; provide an approved source or download instructions when needed.
Setup
Use Python 3.10 or a compatible version supported by the libraries in requirements.txt.
1. Clone or download the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
If the repository is already on your computer, open a terminal at the project root instead.
2. Create and activate a virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
If the app has a separate dependency file, install that too:
pip install -r app/requirements.txt
Run the Notebooks
1. Open the repository in VS Code or Jupyter.
2. Install the dependencies.
3. Confirm the required CSV files are present in data/.
4. Open each notebook in notebooks/.
5. Run every cell from top to bottom and verify there are no errors.
6. Keep the final outputs visible when submitting the notebooks.
Use the same dataset split when comparing supervised models. Fit scalers and encoders on training data only, then apply the fitted transformations to the test data. Use random_state=42 where applicable for reproducibility.
Run the Streamlit Dashboard
From the project root:
streamlit run app/app.py
The dashboard provides pages for Overview, Classification, Regression, Clustering, Dataset Explorer, and Final Decision. Some interactive diagnostic views require the expected CSV files to be available in data/. Verify that the dashboard uses the final notebook results and trained model artifacts before presenting it as a live prediction application.
Results and Interpretation
When reporting results:
- Compare supervised models using the same held-out test set.
- For classification, report Accuracy, Precision, Recall, weighted F1, confusion matrices, and multiclass ROC-AUC (OvR).
- For regression, report R², RMSE, MAE, and 5-fold cross-validation for the two best models.
- For clustering, report all three required internal metrics and interpret them alongside cluster sizes, visualizations, and cluster profiles. A single metric should not be used as the only reason for selecting a clustering result.
- Explain why the final model or clustering configuration was selected and acknowledge limitations.
Reproducibility and GitHub
- Keep notebooks organized with Markdown explanations and clear section headings.
- Run notebooks from top to bottom before committing.
- Record dependencies in requirements.txt (and app/requirements.txt if used).
- Commit work in meaningful stages, for example: data preparation, baseline models, evaluation, clustering visualizations, and dashboard improvements.
- Update this README when datasets, notebook names, results, or setup instructions change.
Acknowledgements
Generative AI tools were used for code scaffolding and dashboard/README drafting. The project team should review, validate, and own the final code, results, interpretations, and conclusions, and cite AI assistance according to course requirements.
