from fastapi import APIRouter, UploadFile, File
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

router = APIRouter()


@router.post("/linear regression")
def perform_linear_regression(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if file.filename.endswith(".csv"):
        # Read the CSV file using pandas
        df = pd.read_csv(file.file)

        # Check if the dependent column is at the end of the DataFrame
        dependent_column = df.columns[-1]

        # Separate independent and dependent variables
        X = df.iloc[:, :-1]
        y = df[dependent_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate the accuracy (R-squared score) of the model
        accuracy = r2_score(y_test, y_pred)

        # Convert accuracy to percentage format
        accuracy_percentage = round(accuracy * 100, 2)

        return {f"accuracy: {accuracy_percentage} %"}

    return {"error": "Uploaded file is not a CSV file"}





@router.post("/classification")
def perform_classification(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if file.filename.endswith(".csv"):
        # Read the CSV file using pandas
        df = pd.read_csv(file.file)

        # Check if the dependent column is at the end of the DataFrame
        target_column = df.columns[-1]

        # Separate features and target variable
        X = df.iloc[:, :-1]
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the Random Forest classifier
        model_classifier = RandomForestClassifier(random_state=42)
        model_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = model_classifier.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)

        # Convert accuracy to percentage format
        accuracy_percentage = round(accuracy * 100, 2)

        return {f"accuracy: {accuracy_percentage} %"}

    return {"error": "Uploaded file is not a CSV file"}



@router.post("/naive_bayes")
def perform_naive_bayes(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if file.filename.endswith(".csv"):
        # Read the CSV file using pandas
        df = pd.read_csv(file.file)

        # Check if the dependent column is at the end of the DataFrame
        target_column = df.columns[-1]

        # Separate features and target variable
        X = df.iloc[:, :-1]
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the Naive Bayes classifier
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)

        # Make predictions
        y_pred = model_nb.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)

        # Convert accuracy to percentage format with two decimal places
        accuracy_percentage = round(accuracy * 100, 2)

        return {f"accuracy: {accuracy_percentage} %"}

    return {"error": "Uploaded file is not a CSV file"}


