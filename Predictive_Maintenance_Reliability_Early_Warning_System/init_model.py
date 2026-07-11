# init_model.py
import pandas as pd
from utils import verify_and_download_dataset
from training import run_training_pipeline

if __name__ == "__main__":
    # This automatically downloads the dataset and runs the CV optimization
    print("Initializing industrial data source...")
    data_path = verify_and_download_dataset()
    
    print("Running automated model training & threshold optimization...")
    df = pd.read_csv(data_path)
    run_training_pipeline(df)
    print("Artifact pipeline saved successfully to models/ directory.")