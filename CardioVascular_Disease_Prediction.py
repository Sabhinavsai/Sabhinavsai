import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def load_data(filepath):
  df = pd.read_csv(filepath,sep=';')
  return df
def preprocess_data(df):
  X = df.drop(columns=['cardio'])  
  y = df['cardio']
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled, y
def plot_target_distribution(y):
  plt.figure(figsize=(6,4))
  sns.countplot(x=y, palette='Set2')
  plt.title("Target Distribution")
  plt.show()
def classification_models(X, y):
  # Plot target distribution
  plot_target_distribution(y)
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # Logistic Regression
  log_model = LogisticRegression(max_iter=1000)
  log_model.fit(X_train, y_train)
  y_pred_log = log_model.predict(X_test)
  print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
  print(classification_report(y_test, y_pred_log))
  
  # Random Forest Classifier
  rf_model = RandomForestClassifier(n_estimators=100)
  rf_model.fit(X_train, y_train)
  y_pred_rf = rf_model.predict(X_test)
  print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
  print(classification_report(y_test, y_pred_rf))
  
  # Gradient Boosting Classifier
  gb_model = GradientBoostingClassifier(n_estimators=100)
  gb_model.fit(X_train, y_train)
  y_pred_gb = gb_model.predict(X_test)
  print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
  print(classification_report(y_test, y_pred_gb))
  
  # Plot feature importance for Random Forest
  feature_importances = rf_model.feature_importances_
  plt.figure(figsize=(10,6))
  sns.barplot(x=feature_importances, y=df.columns[:-1], palette='Blues')
  plt.title("Feature Importance (Random Forest)")
  plt.show()
df = load_data('/content/cardio_train.csv')
X, y = preprocess_data(df)
classification_models(X, y)
