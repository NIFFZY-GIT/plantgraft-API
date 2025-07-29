# main.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time
import logging
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

# --- 0. Professional Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration & File Paths ---
PERSISTENT_DATA_PATH = os.getenv('RENDER_DISK_PATH', '.')

MODEL_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'graft_compatibility_model.keras')
PLANT_INFO_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'plant_lookup_table.csv')
PLANT_NAME_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'plant_name_map.pkl')
FAMILY_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'family_map.pkl')
GENUS_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'genus_map.pkl')

# --- 2. Global Variables to hold loaded artifacts ---
model: keras.Model | None = None
plant_info_df: pd.DataFrame | None = None
plant_name_map: Dict[str, int] | None = None
family_map: Dict[str, int] | None = None
genus_map: Dict[str, int] | None = None
artifacts_loaded_successfully: bool = False

# --- 3. Pydantic Models for Request and Response ---
class GraftablePlantResponse(BaseModel):
    scientific_name: str = Field(..., alias="Scientific Name")
    common_name: str = Field(..., alias="Common Name")
    family: str = Field(..., alias="Family")
    genus: str = Field(..., alias="Genus")
    predicted_compatibility_percent: float = Field(..., alias="Predicted Compatibility (%)")
    class Config: populate_by_name = True

class PredictionResult(BaseModel):
    target_plant_scientific_name: str
    search_time_seconds: float
    graftable_plants_found: int
    results: List[GraftablePlantResponse]

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    artifacts_status: str
    model_loaded: bool
    plant_info_loaded: bool
    plant_name_map_loaded: bool
    family_map_loaded: bool
    genus_map_loaded: bool

class PlantMetadata(BaseModel):
    scientific_name: str = Field(..., description="The scientific name, which will be the primary key.")
    common_name: str | None = Field("N/A", description="The common name of the plant.")
    family: str | None = Field("<UNK>", description="The family of the plant.")
    genus: str | None = Field("<UNK>", description="The genus of the plant.")

# --- 4. Lifespan Event Handler (Unchanged) ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model, plant_info_df, plant_name_map, family_map, genus_map, artifacts_loaded_successfully
    logging.info("Lifespan: Loading model and artifacts...")
    start_load_time = time.time()
    artifacts_loaded_successfully = True

    # Code to load artifacts... (omitted for brevity, no changes needed)
    try:
        model = keras.models.load_model(MODEL_SAVE_PATH)
        logging.info(f"Successfully loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        artifacts_loaded_successfully = False

    try:
        plant_info_df = pd.read_csv(PLANT_INFO_SAVE_PATH, index_col='Name')
        for col in ['Family', 'Genus', 'Common_Name']:
            plant_info_df[col] = plant_info_df[col].fillna('<UNK>')
        logging.info(f"Successfully loaded and preprocessed plant info from {PLANT_INFO_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Error loading plant info: {e}", exc_info=True)
        artifacts_loaded_successfully = False

    maps_to_load = {
        "plant_name_map": PLANT_NAME_MAP_SAVE_PATH,
        "family_map": FAMILY_MAP_SAVE_PATH,
        "genus_map": GENUS_MAP_SAVE_PATH,
    }
    loaded_maps = {}
    for name, path in maps_to_load.items():
        try:
            with open(path, 'rb') as f: loaded_maps[name] = pickle.load(f)
            logging.info(f"Successfully loaded {name} from {path}")
        except Exception as e:
            logging.error(f"Error loading {name}: {e}", exc_info=True)
            artifacts_loaded_successfully = False
    
    if artifacts_loaded_successfully:
        plant_name_map, family_map, genus_map = loaded_maps.get("plant_name_map"), loaded_maps.get("family_map"), loaded_maps.get("genus_map")
        if not all([plant_name_map, family_map, genus_map]):
             logging.critical("FATAL: One or more mapping dictionaries are None after loading attempt.")
             artifacts_loaded_successfully = False

    load_duration = time.time() - start_load_time
    logging.info(f"Lifespan: Artifact loading finished in {load_duration:.2f} seconds.")
    if not artifacts_loaded_successfully:
        logging.critical("CRITICAL WARNING: API started but artifacts failed to load. Endpoints will fail.")

    yield
    logging.info("Lifespan: Application shutting down...")


# --- 5. FastAPI App Initialization ---
app = FastAPI(title="Graft Compatibility API", version="1.3.0-robust", lifespan=lifespan)

# --- 6. Prediction Logic and Other Endpoints (Unchanged) ---
# Your existing /predict_graftable, /health, etc. code goes here. No changes needed.
# Omitting for brevity.

@app.get("/predict_graftable", response_model=PredictionResult, summary="Predict Graftable Plants", tags=["Prediction"])
async def predict_graftable_endpoint(
    scientific_name: str = Query(..., description="Scientific name of the target plant."),
    threshold: float = Query(60.0, ge=0, le=100, description="Minimum compatibility score (0-100).")
):
    # This function is unchanged. Omitting its body.
    # ... (body of find_graftable_plants_batched_for_api logic)
    # The existing function logic is correct.
    pass # Placeholder for your existing code

# --- 7. ROBUST ENDPOINT: Add Plant Metadata ---
@app.post("/plants/add_metadata", status_code=status.HTTP_201_CREATED, summary="Add New Plant Metadata", tags=["Data Management"])
async def add_plant_metadata(data: PlantMetadata):
    """
    Adds a new plant's metadata to the plant_lookup_table.csv. This version has robust error handling.
    """
    logging.info(f"Received request to add metadata for: {data.scientific_name}")
    logging.info(f"Target file path on disk: {PLANT_INFO_SAVE_PATH}")
    
    if not os.path.exists(PLANT_INFO_SAVE_PATH):
        logging.error(f"FATAL: The plant lookup table was not found at {PLANT_INFO_SAVE_PATH}. Cannot proceed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration Error: The plant lookup table file was not found on the server."
        )

    def append_to_csv():
        try:
            if not os.access(PLANT_INFO_SAVE_PATH, os.W_OK):
                logging.error(f"PERMISSION DENIED: Cannot write to {PLANT_INFO_SAVE_PATH}.")
                raise PermissionError(f"Server does not have permission to write to the data file at {PLANT_INFO_SAVE_PATH}")

            logging.info("Reading existing CSV to check for duplicates...")
            df = pd.read_csv(PLANT_INFO_SAVE_PATH, index_col='Name')
            logging.info("Successfully read the existing CSV.")
            
            if data.scientific_name.lower() in [idx.lower() for idx in df.index]:
                logging.info(f"Plant metadata for '{data.scientific_name}' already exists.")
                return "already_exists"

            logging.info(f"'{data.scientific_name}' is a new plant. Preparing to append.")
            new_row = pd.DataFrame([{
                'Name': data.scientific_name,
                'Family': data.family or '<UNK>',
                'Genus': data.genus or '<UNK>',
                'Common_Name': data.common_name or 'N/A'
            }]).set_index('Name')

            new_row.to_csv(PLANT_INFO_SAVE_PATH, mode='a', header=False)
            logging.info(f"Successfully appended new data for {data.scientific_name}")
            return "added_successfully"
        except Exception as e:
            logging.error(f"EXCEPTION during file operation in append_to_csv: {e}", exc_info=True)
            raise e

    try:
        result = await run_in_threadpool(append_to_csv)
        if result == "already_exists":
            return {"message": "Plant metadata already exists.", "scientific_name": data.scientific_name}
        
        return {"message": "Plant metadata added successfully.", "data_added": data.model_dump()}
    except Exception as e:
        error_type = type(e).__name__
        logging.error(f"Raising HTTPException due to caught exception of type {error_type}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred while saving data. Type: {error_type}. Message: {str(e)}"
        )

# Add your other endpoints like /health, / etc. here
@app.get("/", summary="Root Endpoint", tags=["Utility"])
async def read_root():
    status_msg = "API is running and artifacts are loaded." if artifacts_loaded_successfully else "API is running, but critical artifacts failed to load."
    return {"message": "Welcome to the Graft Compatibility API!", "status": status_msg, "docs_url": "/docs"}