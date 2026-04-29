

def selected_feature(feature_importance,x_train, x_test):
# Sort feature importance first
 feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
  ).reset_index(drop=True)

# Select top N
 top_n = 20

 top_features = feature_importance['Feature'].head(top_n).tolist()

# Keep only columns that exist
 top_features = [col for col in top_features if col in x_train.columns]

 print("Top Selected Features:")
 print(top_features)

# Reduce dataset
 x_train_selected = x_train[top_features]
 x_test_selected = x_test[top_features]

 print("New x_train shape:", x_train_selected.shape)
 print("New x_test shape:", x_test_selected.shape)
 return (x_train_selected, x_test_selected)