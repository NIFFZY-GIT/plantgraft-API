# main.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time
import logging
import csv
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

# --- 0. Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. File Paths ---
PERSISTENT_DATA_PATH = os.getenv('RENDER_DISK_PATH', '.')
MODEL_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'graft_compatibility_model.keras')
PLANT_INFO_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'plant_lookup_table.csv')
PLANT_NAME_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'plant_name_map.pkl')
FAMILY_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'family_map.pkl')
GENUS_MAP_SAVE_PATH = os.path.join(PERSISTENT_DATA_PATH, 'genus_map.pkl')

# --- 2. Global Variables ---
model: keras.Model | None = None
plant_info_df: pd.DataFrame | None = None
plant_name_map: Dict[str, int] | None = None
family_map: Dict[str, int] | None = None
genus_map: Dict[str, int] | None = None
artifacts_loaded_successfully: bool = False

# --- 3. Pydantic Models ---
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

# --- 4. Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model, plant_name_map, family_map, genus_map, artifacts_loaded_successfully
    logging.info("Lifespan: Loading model and artifacts...")
    artifacts_loaded_successfully = True

    try:
        model = keras.models.load_model(MODEL_SAVE_PATH)
        logging.info(f"Successfully loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        artifacts_loaded_successfully = False

    maps_to_load = { "plant_name_map": PLANT_NAME_MAP_SAVE_PATH, "family_map": FAMILY_MAP_SAVE_PATH, "genus_map": GENUS_MAP_SAVE_PATH, }
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
    
    yield
    logging.info("Lifespan: Application shutting down...")

# --- 5. FastAPI App Initialization ---
app = FastAPI(title="Graft Compatibility API", version="1.6.0-corrected", lifespan=lifespan)

# --- 6. PREDICTION LOGIC AND ENDPOINT (CORRECTED) ---
@app.get("/predict_graftable/{scientific_name}", response_model=PredictionResult, summary="Predict Graftable Plants", tags=["Prediction"])
async def predict_graftable_endpoint(
    scientific_name: str, # This is now a PATH parameter
    threshold: float = Query(60.0, ge=0, le=100, description="Minimum compatibility score (0-100).")
):
    if not artifacts_loaded_successfully:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server not ready: Critical artifacts not loaded.")

    api_start_time = time.time()
    
    try:
        current_plant_info_df = pd.read_csv(PLANT_INFO_SAVE_PATH, index_col='Name')
    except FileNotFoundError:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Plant lookup table file not found on server.")

    # FastAPI automatically URL-decodes path parameters.
    try:
        target_info = current_plant_info_df.loc[scientific_name]
        target_family = str(target_info['Family'])
        target_genus_enc = genus_map.get(str(target_info['Genus']), genus_map.get('<UNK>'))
        target_plant_name_enc = plant_name_map.get(scientific_name, plant_name_map.get('<UNK>'))
        target_family_enc = family_map.get(target_family, family_map.get('<UNK>'))
    except KeyError:
        logging.warning(f"Target plant '{scientific_name}' not found in database for prediction.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plant '{scientific_name}' not found in our database.")

    partner_plants_df = current_plant_info_df[ (current_plant_info_df.index != scientific_name) & (current_plant_info_df['Family'] == target_family) ]

    if partner_plants_df.empty:
        return { "target_plant_scientific_name": scientific_name, "search_time_seconds": round(time.time() - api_start_time, 4), "graftable_plants_found": 0, "results": [] }

    num_partners = len(partner_plants_df)
    input_data = {
        'plant_a_name': np.full(num_partners, target_plant_name_enc, dtype=np.int32),
        'a_family': np.full(num_partners, target_family_enc, dtype=np.int32),
        'a_genus': np.full(num_partners, target_genus_enc, dtype=np.int32),
        'plant_b_name': np.array([plant_name_map.get(name, plant_name_map.get('<UNK>')) for name in partner_plants_df.index], dtype=np.int32),
        'b_family': np.array([family_map.get(str(fam), family_map.get('<UNK>')) for fam in partner_plants_df['Family'].values], dtype=np.int32),
        'b_genus': np.array([genus_map.get(str(gen), genus_map.get('<UNK>')) for gen in partner_plants_df['Genus'].values], dtype=np.int32)
    }
    
    # Model likely predicts a score from 0.0 to 1.0
    all_predictions = model.predict(input_data, batch_size=32, verbose=0).flatten()
    
    results_list = []
    for i, score in enumerate(all_predictions):
        compatibility_percent = score * 100
        if compatibility_percent > threshold:
            results_list.append(
                GraftablePlantResponse(
                    **{
                        "Scientific Name": partner_plants_df.index[i],
                        "Common Name": str(partner_plants_df.iloc[i].get('Common_Name', 'N/A')),
                        "Family": str(partner_plants_df.iloc[i].get('Family', '<UNK>')),
                        "Genus": str(partner_plants_df.iloc[i].get('Genus', '<UNK>')),
                        "Predicted Compatibility (%)": round(float(compatibility_percent), 2)
                    }
                )
            )
    
    results_list.sort(key=lambda x: x.predicted_compatibility_percent, reverse=True)
    return { "target_plant_scientific_name": scientific_name, "search_time_seconds": round(time.time() - api_start_time, 4), "graftable_plants_found": len(results_list), "results": results_list }

# --- 7. UTILITY AND DATA MANAGEMENT ENDPOINTS ---
@app.get("/", summary="Root Endpoint", tags=["Utility"], include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Graft Compatibility API!", "status": "API is running.", "docs_url": "/docs"}

@app.get("/health", response_model=HealthCheckResponse, summary="Health Check", tags=["Utility"])
async def health_check_endpoint():
    return HealthCheckResponse(status="healthy" if artifacts_loaded_successfully else "unhealthy", message="API is operational." if artifacts_loaded_successfully else "API not fully operational: artifacts missing.", artifacts_status="Loaded" if artifacts_loaded_successfully else "Not Loaded", model_loaded=model is not None, plant_info_loaded=os.path.exists(PLANT_INFO_SAVE_PATH), plant_name_map_loaded=plant_name_map is not None, family_map_loaded=family_map is not None, genus_map_loaded=genus_map is not None)

@app.post("/plants/add_metadata", status_code=status.HTTP_201_CREATED, summary="Add New Plant Metadata", tags=["Data Management"])
async def add_plant_metadata(data: PlantMetadata):
    def efficient_append_and_check():
        try:
            # Create file with header if it doesn't exist
            if not os.path.exists(PLANT_INFO_SAVE_PATH):
                 with open(PLANT_INFO_SAVE_PATH, mode='w', encoding='utf-8', newline='') as f:
                    csv.writer(f).writerow(['Name', 'Family', 'Genus', 'Common_Name'])
            
            # More robust duplicate check
            with open(PLANT_INFO_SAVE_PATH, mode='r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                for row in reader:
                    if row and row[0].strip().lower() == data.scientific_name.strip().lower():
                        return "already_exists"
        except Exception as e:
            raise IOError(f"Failed to read data file for duplicate check: {e}")
        try:
            new_row = [data.scientific_name, data.family or '<UNK>', data.genus or '<UNK>', data.common_name or 'N/A']
            with open(PLANT_INFO_SAVE_PATH, mode='a', encoding='utf-8', newline='') as f:
                csv.writer(f).writerow(new_row)
            return "added_successfully"
        except Exception as e:
            raise IOError(f"Failed to write to data file: {e}")

    try:
        result = await run_in_threadpool(efficient_append_and_check)
        if result == "already_exists":
            # Return 200 OK instead of an error if it already exists, as the client's goal is met.
            return {"message": "Plant metadata already exists.", "scientific_name": data.scientific_name}
        return {"message": "Plant metadata added successfully.", "data_added": data.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred: {str(e)}")