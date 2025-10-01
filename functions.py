from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def replace_nan_with_median(dataframe, column_name):
    median_value = dataframe[column_name].median()
    dataframe[column_name].fillna(median_value, inplace=True)
    
def remove_values_above_max(dataframe, column_name, max_value):
    dataframe.drop(dataframe[dataframe[column_name] > max_value].index, inplace=True)
    
def transform_values(dataframe, column_name, value_mapping):
    dataframe[column_name] = dataframe[column_name].map(value_mapping)

def evaluate_classifier(X_train, y_train, X_test, y_test, model, target_names=['Not Churn', 'Churn']):
    """
    Trains and evaluates a classifier, printing the classification report and AUC-ROC.
    The target names are set to 'Not Churn' and 'Churn' by default.

    Args:
        X_train: Training data (features).
        y_train: Training labels.
        X_test: Test data (features).
        y_test: Test labels.
        model: Classifier (required).
        target_names: List of target names (optional). Defaults to ['Not Churn', 'Churn'].
    """

    model.fit(X_train, y_train)

    # Evaluation on the training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    auc_roc_train = roc_auc_score(y_train, y_train_proba)

    print("----- Training Set -----")
    print(classification_report(y_train, y_train_pred, target_names=target_names))
    print(f"AUC-ROC (Train): {auc_roc_train}")


    # Evaluation on the test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    auc_roc_test = roc_auc_score(y_test, y_test_proba)

    print("\n\n----- Test Set -----")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    print(f"AUC-ROC (Test): {auc_roc_test}") 

def create_and_train_nn_model(X_train, y_train, X_test, y_test): 
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['recall'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    print(classification_report(y_test, y_pred_classes))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred))