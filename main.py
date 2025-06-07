import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time
import logging # Replaced print with the logging module
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# --- 0. Professional Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration & File Paths ---
MODEL_SAVE_PATH = 'graft_compatibility_model.keras'
PLANT_INFO_SAVE_PATH = 'plant_lookup_table.csv'
PLANT_NAME_MAP_SAVE_PATH = 'plant_name_map.pkl'
FAMILY_MAP_SAVE_PATH = 'family_map.pkl'
GENUS_MAP_SAVE_PATH = 'genus_map.pkl'

# --- 2. Global Variables to hold loaded artifacts ---
model: keras.Model | None = None
plant_info_df: pd.DataFrame | None = None
plant_name_map: Dict[str, int] | None = None
family_map: Dict[str, int] | None = None
genus_map: Dict[str, int] | None = None
artifacts_loaded_successfully: bool = False

# --- 3. Pydantic Models for Request and Response (Unchanged) ---
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

class BasicStatsResponse(BaseModel):
    total_known_plants: int
    plant_name_map_size: int
    family_map_size: int
    genus_map_size: int

# --- 4. Lifespan Event Handler for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model, plant_info_df, plant_name_map, family_map, genus_map, artifacts_loaded_successfully
    logging.info("Lifespan: Loading model and artifacts...")
    start_load_time = time.time()
    artifacts_loaded_successfully = True

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
app = FastAPI(title="Graft Compatibility API", version="1.1.0-optimized", lifespan=lifespan)

# --- 6. OPTIMIZED Prediction Logic ---
def find_graftable_plants_batched_for_api(
    target_plant_sci_name: str,
    compatibility_threshold: float = 60.0,
    batch_size: int = 32
) -> Dict[str, Any]:
    if not artifacts_loaded_successfully:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server not ready: Critical artifacts not loaded.")

    api_start_time = time.time()
    target_plant_sci_name = str(target_plant_sci_name)

    unk_plant_name_enc = plant_name_map.get('<UNK>')
    unk_family_enc = family_map.get('<UNK>')
    unk_genus_enc = genus_map.get('<UNK>')
    
    try:
        target_info = plant_info_df.loc[target_plant_sci_name]
        target_family = str(target_info['Family'])
        target_genus_enc = genus_map.get(str(target_info['Genus']), unk_genus_enc)
        target_plant_name_enc = plant_name_map.get(target_plant_sci_name, unk_plant_name_enc)
        target_family_enc = family_map.get(target_family, unk_family_enc)
    except KeyError:
        logging.warning(f"Target plant '{target_plant_sci_name}' not found in database.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plant '{target_plant_sci_name}' not found in our database.")

    # --- KEY OPTIMIZATION: Filter partners by the SAME FAMILY before prediction ---
    partner_plants_df = plant_info_df[
        (plant_info_df.index != target_plant_sci_name) & 
        (plant_info_df['Family'] == target_family)
    ]

    if partner_plants_df.empty:
        logging.info(f"No potential graft partners found in the same family for '{target_plant_sci_name}'.")
        return {
            "target_plant_scientific_name": target_plant_sci_name,
            "search_time_seconds": round(time.time() - api_start_time, 4),
            "graftable_plants_found": 0, "results": []
        }

    logging.info(f"Found {len(partner_plants_df)} potential partners in family '{target_family}' for '{target_plant_sci_name}'.")
    
    # --- The rest of the logic now runs on a much smaller dataset ---
    num_partners = len(partner_plants_df)
    plant_a_name_batch = np.full(num_partners, target_plant_name_enc, dtype=np.int32)
    a_family_batch = np.full(num_partners, target_family_enc, dtype=np.int32)
    a_genus_batch = np.full(num_partners, target_genus_enc, dtype=np.int32)

    partner_sci_names = partner_plants_df.index.to_numpy()
    plant_b_name_batch = np.array([plant_name_map.get(name, unk_plant_name_enc) for name in partner_sci_names], dtype=np.int32)
    b_family_batch = np.array([family_map.get(str(fam), unk_family_enc) for fam in partner_plants_df['Family'].values], dtype=np.int32)
    b_genus_batch = np.array([genus_map.get(str(gen), unk_genus_enc) for gen in partner_plants_df['Genus'].values], dtype=np.int32)

    input_data = {
        'plant_a_name': plant_a_name_batch, 'a_family': a_family_batch, 'a_genus': a_genus_batch,
        'plant_b_name': plant_b_name_batch, 'b_family': b_family_batch, 'b_genus': b_genus_batch
    }
    
    all_predictions = model.predict(input_data, batch_size=batch_size, verbose=0).flatten()
    
    results_list = [
        GraftablePlantResponse(
            **{"Scientific Name": partner_sci_names[i], 
               "Common Name": str(partner_plants_df.iloc[i].get('Common_Name', 'N/A')),
               "Family": str(partner_plants_df.iloc[i].get('Family', '<UNK>')),
               "Genus": str(partner_plants_df.iloc[i].get('Genus', '<UNK>')),
               "Predicted Compatibility (%)": round(float(score), 2)}
        )
        for i, score in enumerate(all_predictions) if score > compatibility_threshold
    ]
    
    results_list.sort(key=lambda x: x.predicted_compatibility_percent, reverse=True)
    total_search_time = time.time() - api_start_time
    return {
        "target_plant_scientific_name": target_plant_sci_name,
        "search_time_seconds": round(total_search_time, 4),
        "graftable_plants_found": len(results_list),
        "results": results_list
    }


# --- 7. API Endpoints (Largely unchanged, but now more robust) ---
@app.get("/", summary="Root Endpoint")
async def read_root():
    status_msg = "API is running and artifacts are loaded." if artifacts_loaded_successfully else "API is running, but critical artifacts failed to load."
    return {"message": "Welcome to the Graft Compatibility API!", "status": status_msg, "docs_url": "/docs"}

@app.get("/health", response_model=HealthCheckResponse, summary="Health Check", tags=["Utility"])
async def health_check_endpoint():
    is_healthy = artifacts_loaded_successfully
    response_data = HealthCheckResponse(
        status="healthy" if is_healthy else "unhealthy",
        message="API is operational." if is_healthy else "API not fully operational: artifacts missing.",
        artifacts_status="All critical artifacts loaded successfully." if is_healthy else "One or more critical artifacts failed to load.",
        model_loaded=model is not None,
        plant_info_loaded=plant_info_df is not None,
        plant_name_map_loaded=plant_name_map is not None,
        family_map_loaded=family_map is not None,
        genus_map_loaded=genus_map is not None
    )
    if is_healthy:
        return response_data
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=response_data.model_dump())

@app.get("/predict_graftable", response_model=PredictionResult, summary="Predict Graftable Plants", tags=["Prediction"])
async def predict_graftable_endpoint(
    scientific_name: str = Query(..., description="Scientific name of the target plant."),
    threshold: float = Query(60.0, ge=0, le=100, description="Minimum compatibility score (0-100).")
):
    try:
        return find_graftable_plants_batched_for_api(
            target_plant_sci_name=scientific_name,
            compatibility_threshold=threshold
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error during prediction for '{scientific_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred during prediction.")

# (Other utility endpoints like /ping, /stats, /known_plants can remain the same as your previous code)