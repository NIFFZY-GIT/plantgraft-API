# Graft Compatibility API — User Manual

This user manual explains how to unpack, prepare, run, and operate the Graft Compatibility API distributed as a ZIP file. It is written for Windows users using PowerShell and assumes you will receive the project archive directly (not via Git).

## Table of contents

- Overview
- Quick checklist
- System requirements
- Files included in a ZIP distribution
- Verify and extract the ZIP
- Prepare Python environment (venv)
- Install dependencies
- Set environment variables
- Generate missing mapping files (`*.pkl`) from CSV (if needed)
- Start the API and validate
- API reference (endpoints, inputs, outputs)
- Adding plant metadata
- Troubleshooting and common errors
- Packaging & sharing the ZIP
- Security and privacy notes
- FAQ
- Contact

## Overview

The Graft Compatibility API is a FastAPI service that predicts graft compatibility between plants using a pre-trained Keras/TensorFlow model. The service requires the model file and several lookup artifacts (CSV and pickles) to be present on disk. The API reads these on startup and exposes REST endpoints to check health, make predictions, and manage plant metadata.

This manual walks you through everything from receiving the ZIP to running quick validations and troubleshooting common issues.

## Quick checklist (high level)

1. Receive ZIP and verify checksum (optional but recommended).
2. Extract ZIP to a working folder.
3. Confirm required files are present: model, CSV, pickles, `main.py`, `requirements.txt`.
4. Create & activate Python virtual environment.
5. Install packages with `pip`.
6. Set `RENDER_DISK_PATH` or place artifacts in project root.
7. Start the API using Uvicorn and validate `/health`.

## System requirements

- Windows 10/11 (or Windows Server) with PowerShell.
- Python 3.10 or 3.11 recommended (the repository contains pycache for 3.11).
- At least 4 GB RAM (more recommended for larger models).
- Network access for clients to call the API.

Optional: GPU + compatible drivers if you plan to run a GPU build of TensorFlow.

## Files included in a ZIP distribution

When you receive the ZIP, expect at minimum:

- `main.py` — FastAPI application.
- `requirements.txt` — Python dependencies.
- `plant_lookup_table.csv` — CSV with columns: `Name`, `Family`, `Genus`, `Common_Name`.
- `graft_compatibility_model.keras` — Keras model directory/file.
- `plant_name_map.pkl`, `family_map.pkl`, `genus_map.pkl` — Python pickles mapping strings to integer ids. (If missing, see section: Generate missing mapping files.)
- `README.md` and/or `USER_MANUAL.md` — documentation files.

If any large model files are omitted intentionally, contact the provider for alternative download instructions.

## Verify and extract the ZIP

1. Compute SHA256 checksum (ask provider to give the expected checksum). In PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\project.zip | Format-List
```

2. Create an extraction folder and extract:

```powershell
New-Item -ItemType Directory -Path 'D:\Projects\plantgraft' -Force
Expand-Archive -Path '.\project.zip' -DestinationPath 'D:\Projects\plantgraft' -Force
Set-Location 'D:\Projects\plantgraft'
```

3. List top-level files to confirm:

```powershell
Get-ChildItem -File -Name
```

## Prepare Python environment (venv)

1. Check Python version:

```powershell
python --version
# recommended: 3.10 or 3.11
```

2. Create a virtual environment and activate it:

```powershell
python -m venv .venv
# If activation is blocked by policy, run the following once for the session:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip:

```powershell
pip install --upgrade pip
```

## Install dependencies

Install packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Notes:

- If `requirements.txt` pins `tensorflow`, confirm compatibility with your Python version.
- For GPU support, follow TensorFlow official GPU installation instructions instead of the default CPU wheel.

## Environment variable: `RENDER_DISK_PATH`

The application looks for artifacts at the path pointed to by `RENDER_DISK_PATH`. If this variable is not set, the app defaults to the current working directory (`.`).

Set it for the current PowerShell session (example):

```powershell
$env:RENDER_DISK_PATH = (Get-Location).Path
```

Or set permanently for your user:

```powershell
setx RENDER_DISK_PATH "D:\Projects\plantgraft"
# Open a new PowerShell session for setx to take effect
```

Place `graft_compatibility_model.keras`, `plant_lookup_table.csv`, and the three pkl files inside that folder (or keep them in the project root and leave `RENDER_DISK_PATH` unset).

## Generate missing mapping files (`*.pkl`) from CSV (if maps are missing)

If `plant_name_map.pkl`, `family_map.pkl`, or `genus_map.pkl` are missing, you can build simple integer encoders from `plant_lookup_table.csv`.

Create a file called `build_maps.py` with the following content and run it in the project folder:

```python
import csv
import pickle
from pathlib import Path

csv_path = Path('plant_lookup_table.csv')
if not csv_path.exists():
    raise SystemExit('plant_lookup_table.csv not found in current folder.')

names = {}
families = {}
genera = {}

next_name = 0
next_family = 0
next_genus = 0

with csv_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get('Name')
        family = row.get('Family') or '<UNK>'
        genus = row.get('Genus') or '<UNK>'
        if name and name not in names:
            names[name] = next_name
            next_name += 1
        if family not in families:
            families[family] = next_family
            next_family += 1
        if genus not in genera:
            genera[genus] = next_genus
            next_genus += 1

# Ensure '<UNK>' exists
for m in (names, families, genera):
    if '<UNK>' not in m:
        m['<UNK>'] = max(m.values(), default=-1) + 1

with open('plant_name_map.pkl','wb') as f:
    pickle.dump(names, f)
with open('family_map.pkl','wb') as f:
    pickle.dump(families, f)
with open('genus_map.pkl','wb') as f:
    pickle.dump(genera, f)

print('Maps built: plant_name_map.pkl, family_map.pkl, genus_map.pkl')
```

Run it:

```powershell
python .\build_maps.py
```

Important: The encoder values generated this way will only be consistent if your model was trained using the same encoding scheme. If you don't have the original pickles, model behavior may be incorrect. Request the original pkl files from the provider whenever possible.

## Start the API

Run the app with Uvicorn. From the activated venv:

```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If the server starts successfully, you will see logs indicating attempts to load the model and maps. If the artifacts are present and load correctly, the app will report readiness in `/health`.

Open API docs in a browser:

http://localhost:8000/docs

## Validate the service (smoke tests)

1. Health check:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET | ConvertTo-Json -Depth 5
```

Expected fields: `status`, `model_loaded`, `plant_info_loaded`, `plant_name_map_loaded`, `family_map_loaded`, `genus_map_loaded`.

2. Prediction (replace `Some_Scientific_Name` with a name present in `plant_lookup_table.csv`):

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict_graftable/Some_Scientific_Name?threshold=65" -Method GET | ConvertTo-Json -Depth 5
```

The response includes `results` (high-compatibility matches) and `other_notable_results`.

3. Add a new plant (example):

```powershell
$body = @{
  scientific_name = 'Plantae exampleus'
  common_name = 'Example Plant'
  family = 'Exampleaceae'
  genus = 'Examplea'
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/plants/add_metadata" -Method POST -Body $body -ContentType "application/json"
```

Check that `plant_lookup_table.csv` has the appended row.

## API Reference

All endpoints are defined in `main.py`. Below is a concise reference.

1) GET /health
- Purpose: Check readiness and which artifacts loaded.
- Response: JSON matching `HealthCheckResponse` schema (see source) with boolean flags.

2) GET /predict_graftable/{scientific_name}
- Purpose: Return predicted graftable partner plants for the provided scientific name.
- Path parameter: `scientific_name` (string) — must exactly match an entry in `plant_lookup_table.csv` Name index.
- Query parameter: `threshold` (float, 0–100; default 65.0) — minimum score for high compatibility.
- Response: JSON matching `PredictionResult` with `results` and `other_notable_results`.

3) POST /plants/add_metadata
- Purpose: Append a plant metadata row to `plant_lookup_table.csv`.
- Body: JSON matching `PlantMetadata` schema: `scientific_name` (required), `common_name`, `family`, `genus`.
- Response: `AddPlantResponse` indicating success or 'already_exists'.

Notes about scoring: The application expects the model's prediction output to be in a format matching the code in `main.py`. If predictions are unexpectedly scaled (e.g., 0–1 instead of 0–100), you may need to adjust presentation logic.

## Adding plant metadata safely

- Use the `/plants/add_metadata` endpoint to safely append without concurrent write issues (the endpoint uses an efficient append and duplicate check inside a threadpool worker).
- If you plan to edit the CSV manually, take a backup first. Ensure the CSV header is: `Name,Family,Genus,Common_Name` and that `Name` values exactly match the keys used in the pickles and model data.

## Troubleshooting and common errors

- Model not loaded (`model_loaded=false` in /health)
  - Confirm `graft_compatibility_model.keras` exists in `RENDER_DISK_PATH` or project root.
  - Check Python/TensorFlow compatibility. Mismatched versions commonly cause load errors.
  - Inspect server logs for the exact exception.

- Missing pickles (`*_map.pkl` not loaded)
  - Ensure `plant_name_map.pkl`, `family_map.pkl`, and `genus_map.pkl` are present and readable.
  - If not present, run `build_maps.py` but note encoding consistency warning above.

- Prediction returns 0 results or raises 404 for `scientific_name`
  - Confirm the `scientific_name` is present in `plant_lookup_table.csv` and matches exactly (case and spacing).

- `PermissionError` when writing to CSV
  - Ensure the process has write permission to `plant_lookup_table.csv`. If your file is read-only, clear the attribute or run the process with an account that has write access.

- PowerShell activation blocked
  - Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force` before activating the venv.

## Packaging & sharing the ZIP

To create a ZIP containing everything (including model and pkl files):

```powershell
# from project root
Compress-Archive -Path * -DestinationPath '..\plantgraft-project.zip' -Force
Get-FileHash -Algorithm SHA256 '..\plantgraft-project.zip' | Format-List
```

Supply the ZIP and the SHA256 checksum to recipients. Instruct recipients to verify the checksum after download.

## Security and privacy notes

- Do not include secrets, API keys, or credentials in the ZIP.
- If your model or dataset is sensitive, use secure transfer (SFTP, signed download links, or private cloud storage) and avoid public channels.

## FAQ

Q: The model predictions are between 0 and 1 but code expects 0-100. What to do?

A: Inspect the output of `model.predict(...)` in `main.py`. If values are 0–1, multiply by 100 when building `Predicted Compatibility (%)`. The repository already shows a place where this scaling decision was handled — validate against training code.

Q: I don't have the original `.pkl` files. Can I use `build_maps.py`?

A: You can, but predictions may be incorrect unless the model was trained using the same encodings. Always try to obtain the original `.pkl` files used during model training.

Q: How do I run without installing dependencies globally?

A: Use the included virtual environment steps above. Keep the venv inside the project folder to make distribution simple.

## Contact

If you face any problem, please contact: k.nipuna.dasun@gmail.com

---

If you want, I can also add the following helper artifacts to the ZIP:

- `setup.ps1` — PowerShell script to automate extraction, venv creation, dependency installation, and a health check.
- `build_maps.py` — the generator script (I included the code above; I can add it to the repo if you want).
- `validate_artifacts.py` — a startup validator that returns a clear list of missing files and their sizes.

Tell me which helper artifacts you'd like me to add and I'll create them in the project. 
