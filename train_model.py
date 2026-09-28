import os
import joblib
import pandas as pd
import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = "data/Titanic-Dataset.csv"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]
    ),
    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5
                )
            )
        ]
    ),
    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42
                )
            )
        ]
    ),
    "Gradient Boosting": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42
                )
            )
        ]
    ),
    "XGBoost": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42
                )
            )
        ]
    )
}

results = []
trained_models = {}

print("\n================ MODEL TRAINING ================\n")

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    )

    trained_models[name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("---------------------------------------------")

results_df = pd.DataFrame(results)

print("\n================ ML MODEL COMPARISON ================\n")
print(
    results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).to_string(index=False)
)

print("\nPreparing data for deep learning...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(
    preprocessor,
    "models/titanic_preprocessor.pkl"
)

print("Preprocessor saved successfully!")

X_train_processed = (
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

X_test_processed = (
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

ann = keras.Sequential(
    [
        keras.layers.Input(
            shape=(X_train_processed.shape[1],)
        ),
        keras.layers.Dense(
            64,
            activation="relu"
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            32,
            activation="relu"
        ),
        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining ANN...")

ann.fit(
    X_train_processed,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

ann_prob = ann.predict(
    X_test_processed,
    verbose=0
).ravel()

ann_pred = (
    ann_prob >= 0.5
).astype(int)

ann_accuracy = accuracy_score(
    y_test,
    ann_pred
)

ann_precision = precision_score(
    y_test,
    ann_pred
)

ann_recall = recall_score(
    y_test,
    ann_pred
)

ann_f1 = f1_score(
    y_test,
    ann_pred
)

ann_roc_auc = roc_auc_score(
    y_test,
    ann_prob
)

print("\nANN Results")
print("Accuracy :", round(ann_accuracy, 4))
print("Precision:", round(ann_precision, 4))
print("Recall   :", round(ann_recall, 4))
print("F1 Score :", round(ann_f1, 4))
print("ROC-AUC  :", round(ann_roc_auc, 4))

cnn_train = X_train_processed.reshape(
    X_train_processed.shape[0],
    X_train_processed.shape[1],
    1
)

cnn_test = X_test_processed.reshape(
    X_test_processed.shape[0],
    X_test_processed.shape[1],
    1
)

cnn = keras.Sequential(
    [
        keras.layers.Input(
            shape=(cnn_train.shape[1], 1)
        ),
        keras.layers.Conv1D(
            32,
            3,
            activation="relu",
            padding="same"
        ),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(
            64,
            3,
            activation="relu",
            padding="same"
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(
            32,
            activation="relu"
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)

cnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining 1D CNN...")

cnn.fit(
    cnn_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

cnn_prob = cnn.predict(
    cnn_test,
    verbose=0
).ravel()

cnn_pred = (
    cnn_prob >= 0.5
).astype(int)

cnn_accuracy = accuracy_score(
    y_test,
    cnn_pred
)

cnn_precision = precision_score(
    y_test,
    cnn_pred
)

cnn_recall = recall_score(
    y_test,
    cnn_pred
)

cnn_f1 = f1_score(
    y_test,
    cnn_pred
)

cnn_roc_auc = roc_auc_score(
    y_test,
    cnn_prob
)

print("\n1D CNN Results")
print("Accuracy :", round(cnn_accuracy, 4))
print("Precision:", round(cnn_precision, 4))
print("Recall   :", round(cnn_recall, 4))
print("F1 Score :", round(cnn_f1, 4))
print("ROC-AUC  :", round(cnn_roc_auc, 4))

results_df.loc[len(results_df)] = {
    "Model": "ANN",
    "Accuracy": ann_accuracy,
    "Precision": ann_precision,
    "Recall": ann_recall,
    "F1 Score": ann_f1,
    "ROC-AUC": ann_roc_auc
}

results_df.loc[len(results_df)] = {
    "Model": "1D CNN",
    "Accuracy": cnn_accuracy,
    "Precision": cnn_precision,
    "Recall": cnn_recall,
    "F1 Score": cnn_f1,
    "ROC-AUC": cnn_roc_auc
}

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n================ FINAL MODEL COMPARISON ================\n")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_f1 = results_df.iloc[0]["F1 Score"]

print("\n=================================================")
print("Overall Best Model:", best_model_name)
print("Best F1 Score:", best_f1)
print("=================================================")

results_df.to_csv(
    "models/model_results.csv",
    index=False
)

if best_model_name in trained_models:
    best_model_path = "models/titanic_best_model.pkl"

    joblib.dump(
        trained_models[best_model_name],
        best_model_path
    )

    print("\nBest ML model saved successfully!")
    print("Saved at:", best_model_path)

elif best_model_name == "ANN":
    best_model_path = "models/titanic_best_model.keras"

    ann.save(best_model_path)

    print("\nBest Deep Learning model saved successfully!")
    print("Saved at:", best_model_path)

elif best_model_name == "1D CNN":
    best_model_path = "models/titanic_best_model.keras"

    cnn.save(best_model_path)

    print("\nBest Deep Learning model saved successfully!")
    print("Saved at:", best_model_path)

print("\nPreprocessor:")
print("models/titanic_preprocessor.pkl")

print("\nModel comparison:")
print("models/model_results.csv")

print("\nTraining completed successfully!")