import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def preprocess_data(data_path, test_size=0.2, target_name="price"):
    """
    Loads data, splits into train/test, performs normalization and one-hot encoding,
    saves preprocessed data with targets as CSV files.
    Args:
        data_path: Path to the CSV data file.
        test_size: Proportion of data for the test set (default: 0.2).
        target_name: Name of the target column (default: "price").
    """

    # Read data
    data = pd.read_csv(data_path)

    # Separate features and target
    features = data.drop(target_name, axis=1)
    target = data[[target_name]]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42
    )
    # Create pipelines
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("ordinal", OrdinalEncoder())])

    # Separate numeric and categorical features
    numeric_features = ["x", "y", "z"]

    # Apply pipelines to training data

    X_train_numeric = numeric_pipeline.fit_transform(X_train[numeric_features])

    # Combine preprocessed features
    X_train_numeric = pd.DataFrame(X_train_numeric, columns=numeric_features)

    X_train_preprocessed = X_train_numeric

    # Apply pipelines (without fitting) to testing data

    X_test_numeric = X_train

    X_test_numeric = pd.DataFrame(X_test_numeric, columns=numeric_features)

    X_test_preprocessed = X_test_numeric

    # Combine features and target into single dataframes
    train_data = pd.concat(
        [X_train_preprocessed, y_train.reset_index(drop=True)], axis=1
    )
    test_data = pd.concat([X_test_preprocessed, y_test.reset_index(drop=True)], axis=1)

    # Save preprocessed data with targets
    train_data.to_csv("data/train.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)


# Set data path and run preprocessing
data_path = "data/diamonds.csv"
preprocess_data(data_path)

print("Preprocessing complete! Train and test data with targets saved as CSV files.")
