# Graft Compatibility API

A small FastAPI service that predicts graft compatibility between plants using a pre-trained TensorFlow model and local lookup artifacts.

## Quick overview

- Language: Python
- Web framework: FastAPI (served via Uvicorn)
- ML: TensorFlow / Keras model (`graft_compatibility_model.keras`)

The project expects the trained model and a few lookup files (CSV + pickles) to be present on disk. The app reads those on startup and exposes prediction and utility endpoints.

## Requirements

- Python 3.11 (the project contains pycache for CPython 3.11; other 3.10+ versions may work but 3.11 is recommended)
- Windows PowerShell (instructions below assume PowerShell)

Files the app expects (place these in your working folder or point `RENDER_DISK_PATH` to their folder):

- `graft_compatibility_model.keras` — trained Keras model file
- `plant_lookup_table.csv` — plants lookup CSV (columns: `Name`, `Family`, `Genus`, `Common_Name`)
- `plant_name_map.pkl` — pickle mapping plant names to integer ids
- `family_map.pkl` — pickle mapping family names to integer ids
- `genus_map.pkl` — pickle mapping genus names to integer ids

If any of these artifacts are missing the API will mark itself as not ready and return errors for prediction requests.

## Setup (Windows PowerShell)

1. Open PowerShell and change directory into the project folder:

```powershell
cd 'D:\Campus\UNDERGRADUATE PROJECT\Project\plantgraft-API'
```

2. (Recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
# If your session blocks script execution, run the following once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If `tensorflow` in `requirements.txt` is a CPU-only build it will install without GPU support. To enable GPU support, follow TensorFlow's official GPU installation guide for your OS and GPU.

## Environment variables

The app uses the `RENDER_DISK_PATH` environment variable to locate persistent artifacts. If not set, it defaults to the current working directory (`.`).

To set it for the current PowerShell session:

```powershell
$env:RENDER_DISK_PATH = 'D:\path\to\artifacts'
```

To set it permanently for your user (so it persists across sessions):

```powershell
setx RENDER_DISK_PATH "D:\path\to\artifacts"
# Note: you must start a new shell for setx to take effect
```

Place the required artifact files inside the folder pointed to by `RENDER_DISK_PATH` or the project root.

## Run the API

Start the application using Uvicorn (development mode with auto reload):

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If you don't have `uvicorn` installed globally, use the venv's Python to run it:

```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, open the interactive API docs at:

http://localhost:8000/docs

## Useful endpoints

- Health check: GET `/health` — returns readiness and which artifacts were loaded
- Predict: GET `/predict_graftable/{scientific_name}?threshold=65` — returns predicted graftable partners for the given scientific name
- Add plant metadata: POST `/plants/add_metadata` — add a new CSV row to `plant_lookup_table.csv`

Example GET request (PowerShell):

```powershell
curl.exe -Method GET "http://localhost:8000/predict_graftable/SOME_SCIENTIFIC_NAME?threshold=65"
```

Example POST to add plant metadata (PowerShell):

```powershell
$body = @{
  scientific_name = 'Plantae exampleus'
  common_name = 'Example Plant'
  family = 'Exampleaceae'
  genus = 'Examplea'
} | ConvertTo-Json

curl.exe -Method POST -Uri "http://localhost:8000/plants/add_metadata" -Body $body -ContentType "application/json"
```

## Troubleshooting

- If `/health` shows `model_loaded=false` or `artifacts_status=Not Loaded`, ensure the model and mapping files are present in `RENDER_DISK_PATH` or the project root and that the process has read permissions.
- If model load fails with TensorFlow errors, verify your installed TensorFlow build matches your Python version and OS architecture (and GPU drivers if using GPU).
- If activation of `.venv` fails in PowerShell due to policy, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force` in the shell before activating.

## Development notes

- The main code is in `main.py`.
- Prediction uses the Keras model and expects the pickles to map names/families/genus to integer ids.
- The app will return HTTP 503 if artifacts fail to load at startup.

## Tests and validation

There are no automated tests included. Manual validation steps:

1. Start the API.
2. Check `/health` for artifact load status.
3. Try `/predict_graftable/<some-scientific-name>` for a plant present in `plant_lookup_table.csv`.

## Next steps / improvements

- Add automated unit tests for data-loading and prediction logic.
- Add schema validation for `plant_lookup_table.csv` and helper scripts to build the pickle maps from the CSV.
- Dockerize the service for easier deployment.

## Contact / License

This project is a student/experimental project. No license file is included; add one if you plan to distribute it.

If you face any problem, please contact: k.nipuna.dasun@gmail.com

---

If you want, I can create this `README.md` in the repo directly and also add a short script to validate that required artifact files exist on startup.
