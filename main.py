from typing import Union
import pandas as pd
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.post("/upload/csv/")
def upload_csv(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if file.filename.endswith(".csv"):
        # Read the CSV file using pandas
        df = pd.read_csv(file.file)
        
        # Get the column names as a list
        column_names_list = df.columns.tolist()
        
        # Get the column names as a dictionary
        column_names_dict = {"column_names": column_names_list}
        
        return column_names_dict
    
    return {"error": "Uploaded file is not a CSV file"}