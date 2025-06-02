import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Eksperimen")

# Memuat dataset
DATA_PATH = 'car_df_preprocessing.csv'
car_df = pd.read_csv(DATA_PATH)

# Split dataset
X = car_df.drop(columns=['annual_income_category'])
y = car_df['annual_income_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    mlflow.autolog()
    
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    # Logging model 
    mlflow.sklearn.log_model(model_rf,"model_default")
    
    # Log metrics ke MLflow
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })