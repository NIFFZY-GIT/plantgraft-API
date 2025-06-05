import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager # For lifespan events

# --- 0. Configuration & File Paths ---
MODEL_SAVE_PATH = 'graft_compatibility_model.keras'
PLANT_INFO_SAVE_PATH = 'plant_lookup_table.csv'
PLANT_NAME_MAP_SAVE_PATH = 'plant_name_map.pkl'
FAMILY_MAP_SAVE_PATH = 'family_map.pkl'
GENUS_MAP_SAVE_PATH = 'genus_map.pkl'

# --- Global Variables to hold loaded artifacts ---
model: keras.Model | None = None
plant_info_df: pd.DataFrame | None = None
plant_name_map: Dict[str, int] | None = None
family_map: Dict[str, int] | None = None
genus_map: Dict[str, int] | None = None

# --- Pydantic Models for Request and Response ---
class GraftablePlantResponse(BaseModel):
    scientific_name: str = Field(..., alias="Scientific Name")
    common_name: str = Field(..., alias="Common Name")
    family: str = Field(..., alias="Family")
    genus: str = Field(..., alias="Genus")
    predicted_compatibility_percent: float = Field(..., alias="Predicted Compatibility (%)")

    class Config:
        populate_by_name = True

class PredictionResult(BaseModel):
    target_plant_scientific_name: str
    search_time_seconds: float
    graftable_plants_found: int
    results: List[GraftablePlantResponse]

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model, plant_info_df, plant_name_map, family_map, genus_map
    print("Lifespan: Loading model and artifacts...")
    start_load_time = time.time()
    all_loaded_successfully = True

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model = keras.models.load_model(MODEL_SAVE_PATH)
            print(f"Successfully loaded model from {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            all_loaded_successfully = False
    else:
        print(f"FATAL: Model file not found at {MODEL_SAVE_PATH}")
        all_loaded_successfully = False

    if os.path.exists(PLANT_INFO_SAVE_PATH):
        try:
            plant_info_df = pd.read_csv(PLANT_INFO_SAVE_PATH)
            if 'Name' in plant_info_df.columns:
                plant_info_df.set_index('Name', inplace=True)
                for col in ['Family', 'Genus', 'Common_Name']:
                    if col not in plant_info_df.columns:
                        plant_info_df[col] = '<UNK>' # Initialize column if missing
                    # CORRECTED PANDAS FILLNA:
                    plant_info_df[col] = plant_info_df[col].fillna('<UNK>')
                print(f"Successfully loaded plant info from {PLANT_INFO_SAVE_PATH}")
            else:
                print(f"FATAL: 'Name' column not found in {PLANT_INFO_SAVE_PATH}.")
                all_loaded_successfully = False
        except Exception as e:
            print(f"Error loading plant info: {e}")
            all_loaded_successfully = False
    else:
        print(f"FATAL: Plant info file not found at {PLANT_INFO_SAVE_PATH}")
        all_loaded_successfully = False

    map_files_config = {
        "plant_name_map": PLANT_NAME_MAP_SAVE_PATH,
        "family_map": FAMILY_MAP_SAVE_PATH,
        "genus_map": GENUS_MAP_SAVE_PATH,
    }
    maps_temp = {}
    for map_name_key, path in map_files_config.items(): # Renamed map_name to map_name_key to avoid conflict
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f: maps_temp[map_name_key] = pickle.load(f)
                print(f"Successfully loaded {map_name_key} from {path}")
            except Exception as e:
                print(f"Error loading {map_name_key}: {e}")
                all_loaded_successfully = False
        else:
            print(f"FATAL: {map_name_key} file not found at {path}")
            all_loaded_successfully = False
    
    if all_loaded_successfully:
        plant_name_map = maps_temp.get("plant_name_map")
        family_map = maps_temp.get("family_map")
        genus_map = maps_temp.get("genus_map")
        if not all([plant_name_map, family_map, genus_map]):
             print("FATAL: One or more mapping dictionaries are missing after loading attempt.")
             all_loaded_successfully = False

    load_duration = time.time() - start_load_time
    print(f"Lifespan: Artifact loading finished in {load_duration:.2f} seconds.")
    if not all_loaded_successfully:
        print("CRITICAL WARNING: API started but one or more critical artifacts failed to load. Endpoints will likely fail.")
    
    yield
    print("Lifespan: Application shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Graft Compatibility API",
    description="Predicts graft compatibility between plants.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Prediction Logic ---
def find_graftable_plants_batched_for_api(
    target_plant_sci_name: str,
    compatibility_threshold: float = 60.0,
    batch_size: int = 32
) -> Dict[str, Any]:
    global model, plant_info_df, plant_name_map, family_map, genus_map

    if not all([model, plant_info_df is not None, plant_name_map, family_map, genus_map]):
        raise HTTPException(status_code=503, detail="Server not ready: Critical artifacts not loaded or missing.")

    api_start_time = time.time()
    target_plant_sci_name = str(target_plant_sci_name)

    unk_plant_name_enc = plant_name_map.get('<UNK>')
    unk_family_enc = family_map.get('<UNK>')
    unk_genus_enc = genus_map.get('<UNK>')
    if any(enc is None for enc in [unk_plant_name_enc, unk_family_enc, unk_genus_enc]):
        raise HTTPException(status_code=500, detail="Internal server error: <UNK> token missing in critical maps.")

    target_plant_name_enc = plant_name_map.get(target_plant_sci_name, unk_plant_name_enc)
    try:
        target_info = plant_info_df.loc[target_plant_sci_name]
        target_family_enc = family_map.get(str(target_info['Family']), unk_family_enc)
        target_genus_enc = genus_map.get(str(target_info['Genus']), unk_genus_enc)
    except KeyError:
        target_family_enc = unk_family_enc
        target_genus_enc = unk_genus_enc

    partner_plants_df = plant_info_df[plant_info_df.index != target_plant_sci_name]
    if partner_plants_df.empty:
        return {
            "target_plant_scientific_name": target_plant_sci_name,
            "search_time_seconds": round(time.time() - api_start_time, 4),
            "graftable_plants_found": 0,
            "results": []
        }

    num_partners = len(partner_plants_df)
    plant_a_name_batch = np.full(num_partners, target_plant_name_enc, dtype=np.int32)
    a_family_batch = np.full(num_partners, target_family_enc, dtype=np.int32)
    a_genus_batch = np.full(num_partners, target_genus_enc, dtype=np.int32)

    partner_sci_names = partner_plants_df.index.to_numpy()
    plant_b_name_batch = np.array([plant_name_map.get(name, unk_plant_name_enc) for name in partner_sci_names], dtype=np.int32)
    b_family_batch = np.array([family_map.get(str(fam), unk_family_enc) for fam in partner_plants_df['Family'].values], dtype=np.int32)
    b_genus_batch = np.array([genus_map.get(str(gen), unk_genus_enc) for gen in partner_plants_df['Genus'].values], dtype=np.int32)

    input_data_batched = {
        'plant_a_name': plant_a_name_batch, 'a_family': a_family_batch, 'a_genus': a_genus_batch,
        'plant_b_name': plant_b_name_batch, 'b_family': b_family_batch, 'b_genus': b_genus_batch
    }
    
    all_predictions = model.predict(input_data_batched, batch_size=batch_size, verbose=0)
    all_predictions = np.clip(all_predictions.flatten(), 0, 100)

    results_list: List[GraftablePlantResponse] = []
    for i in range(num_partners):
        score = all_predictions[i]
        if score > compatibility_threshold:
            partner_sci_name = partner_sci_names[i]
            partner_info = partner_plants_df.iloc[i]
            results_list.append(
                GraftablePlantResponse(
                    **{
                        "Scientific Name": partner_sci_name,
                        "Common Name": str(partner_info.get('Common_Name', 'N/A')),
                        "Family": str(partner_info.get('Family', '<UNK>')),
                        "Genus": str(partner_info.get('Genus', '<UNK>')),
                        "Predicted Compatibility (%)": round(score, 2)
                    }
                )
            )
    
    results_list.sort(key=lambda x: x.predicted_compatibility_percent, reverse=True)
    
    total_search_time = time.time() - api_start_time
    return {
        "target_plant_scientific_name": target_plant_sci_name,
        "search_time_seconds": round(total_search_time, 4),
        "graftable_plants_found": len(results_list),
        "results": results_list
    }

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint")
async def read_root():
    artifacts_loaded = all([model, plant_info_df is not None, plant_name_map, family_map, genus_map])
    status_message = "API is running and artifacts appear to be loaded." if artifacts_loaded else "API is running, but one or more critical artifacts might be missing or failed to load."
    return {"message": "Welcome to the Graft Compatibility API!", "status": status_message, "docs_url": "/docs"}

@app.get("/predict_graftable", response_model=PredictionResult, summary="Predict Graftable Plants")
async def predict_graftable_endpoint(
    scientific_name: str = Query(..., min_length=1, description="Scientific name of the target plant."),
    threshold: float = Query(60.0, ge=0, le=100, description="Minimum compatibility score (0-100)."),
    prediction_batch_size: int = Query(32, ge=1, description="Internal batch size for model prediction.")
):
    try:
        prediction_output = find_graftable_plants_batched_for_api(
            target_plant_sci_name=scientific_name,
            compatibility_threshold=threshold,
            batch_size=prediction_batch_size
        )
        return prediction_output
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        # import traceback; print(traceback.format_exc()) # For more detailed server-side logging
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/known_plants", summary="List a Sample of Known Plants")
async def get_known_plants(sample_size: int = Query(10, ge=1, le=100, description="Number of sample plant names.")):
    if plant_info_df is None or plant_info_df.empty:
        raise HTTPException(status_code=503, detail="Plant information not loaded or empty.")
    actual_sample_size = min(sample_size, len(plant_info_df))
    sample_names = plant_info_df.sample(n=actual_sample_size).index.tolist()
    return {"sample_known_scientific_names": sample_names}

# Run with: uvicorn main:app --reload