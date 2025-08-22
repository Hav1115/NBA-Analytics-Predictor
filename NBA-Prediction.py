#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# CSV Load
df = pd.read_csv("nba_games_stats.csv")



df['fg_diff'] = df['home_fg%'] - df['away_fg%']
df['rebounds_diff'] = df['home_rebounds'] - df['away_rebounds']

df['turnovers_diff'] = df['away_turnovers'] - df['home_turnovers']
df['assists_diff'] = df['home_assists'] - df['away_assists']


features = ['fg_diff', 'rebounds_diff', 'turnovers_diff', 'assists_diff']
X = df[features]
y = df['home_win']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)



# Pipeline for Logistic Regression
pipeline_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Pipeline for Random Classifier
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])




param_grid_log = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}


param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5]
}

# Grid search for Logistic Regression
grid_log = GridSearchCV(pipeline_log, param_grid_log, cv=5, scoring='accuracy')
grid_log.fit(X_train, y_train)

# Grid search for Random Forest
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", grid_log.best_params_)
print("Best cross-validation accuracy for Logistic Regression: {:.3f}".format(grid_log.best_score_))
print()
print("Best parameters for Random Forest:", grid_rf.best_params_)
print("Best cross-validation accuracy for Random Forest: {:.3f}".format(grid_rf.best_score_))
print("------------------------------------------------------------")



# Predictions
pred_log = grid_log.predict(X_test)
pred_rf = grid_rf.predict(X_test)

print("\nLogistic Regression Test Accuracy:", accuracy_score(y_test, pred_log))
print("Classification Report for Logistic Regression:\n", classification_report(y_test, pred_log))
print("------------------------------------------------------------")
print("Random Forest Test Accuracy:", accuracy_score(y_test, pred_rf))
print("Classification Report for Random Forest:\n", classification_report(y_test, pred_rf))
print("------------------------------------------------------------")

# Higher test wins
selected_model = grid_log if accuracy_score(y_test, pred_log) >= accuracy_score(y_test, pred_rf) else grid_rf

new_game = pd.DataFrame({
    'home_fg%': [0.47],
    'away_fg%': [0.44],
    'home_rebounds': [50],
    'away_rebounds': [45],
    'home_turnovers': [13],
    'away_turnovers': [16],
    'home_assists': [25],
    'away_assists': [20]
})

new_game['fg_diff'] = new_game['home_fg%'] - new_game['away_fg%']
new_game['rebounds_diff'] = new_game['home_rebounds'] - new_game['away_rebounds']
new_game['turnovers_diff'] = new_game['away_turnovers'] - new_game['home_turnovers']
new_game['assists_diff'] = new_game['home_assists'] - new_game['away_assists']

new_game_features = new_game[features]
prediction_new = selected_model.predict(new_game_features)

print("\nFinal Selected Model Prediction for the new game (1 = home win, 0 = home loss):", prediction_new[0])
