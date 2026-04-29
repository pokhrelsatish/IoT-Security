from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_random_forest(x_train, y_train, x_test, y_test):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,          #use all CPU Cores
        max_depth=20,       #prevents overfitting
        min_samples_split=5,
        min_samples_leaf=2
    )

    # Train model
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)

    print("Model trained successfully")
    
    # Feature importance
    feature_importance = pd.DataFrame({
       "Feature": x_train.columns,
       "Importance": model.feature_importances_
   }).sort_values(by="Importance", ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    return model, y_pred, feature_importance

