import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle # For saving mapping dictionaries
import os # For path definitions

# --- 0. Configuration & File Paths ---
CSV_FILE_PATH = 'plant_compatibility.csv' # Path to your new CSV data
PLANT_INFO_SAVE_PATH = 'plant_lookup_table.csv' # For saving the comprehensive plant info
PLANT_NAME_MAP_SAVE_PATH = 'plant_name_map.pkl'
FAMILY_MAP_SAVE_PATH = 'family_map.pkl'
GENUS_MAP_SAVE_PATH = 'genus_map.pkl'
MODEL_SAVE_PATH = 'graft_compatibility_model.keras'

# --- 1. Data Preparation ---

# Define expected columns (these must be in your input CSV file)
EXPECTED_COLUMNS_IN_FILE = [
    'Plant_A_Name', 'A_Family', 'A_Genus', 'A_Common_Name',
    'Plant_B_Name', 'B_Family', 'B_Genus', 'B_Common_Name',
    'Compatibility (%)' # This is the specific name expected in the CSV for the target
]
# This is the name the target column will have *internally* after loading and potential renaming.
TARGET_COLUMN_NAME = 'Compatibility'

# Load the dataset
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded {CSV_FILE_PATH}")
    print("DataFrame head:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print("\nDataFrame columns found in CSV:", df.columns.tolist())

    # Validate columns found in the file
    missing_cols = [col for col in EXPECTED_COLUMNS_IN_FILE if col not in df.columns]
    critical_input_features = [col for col in EXPECTED_COLUMNS_IN_FILE if col != 'Compatibility (%)']


    if missing_cols:
        # If 'Compatibility (%)' is missing, but 'Compatibility' (our internal target name) is present
        if 'Compatibility (%)' in missing_cols and TARGET_COLUMN_NAME in df.columns:
            print(f"Found '{TARGET_COLUMN_NAME}' column. Script expected '{EXPECTED_COLUMNS_IN_FILE[-1]}' in CSV but will use existing '{TARGET_COLUMN_NAME}'.")
        # If other critical columns are missing (not just a target name variation)
        elif any(col in missing_cols for col in critical_input_features):
             raise ValueError(f"CSV file is missing one or more required feature columns: {[col for col in missing_cols if col != 'Compatibility (%)']}. \nExpected columns in file are: {EXPECTED_COLUMNS_IN_FILE}")
        # If 'Compatibility (%)' is missing and 'Compatibility' is also missing.
        elif 'Compatibility (%)' in missing_cols and TARGET_COLUMN_NAME not in df.columns:
             raise ValueError(f"Target column missing. Expected '{EXPECTED_COLUMNS_IN_FILE[-1]}' or '{TARGET_COLUMN_NAME}' in CSV.")
        else: # Catch-all for other missing column combinations
            raise ValueError(f"CSV file is missing the following required columns: {missing_cols}. \nExpected columns in file are: {EXPECTED_COLUMNS_IN_FILE}")

    # If all expected columns (including 'Compatibility (%)') are present, rename 'Compatibility (%)'
    elif EXPECTED_COLUMNS_IN_FILE[-1] in df.columns and EXPECTED_COLUMNS_IN_FILE[-1] != TARGET_COLUMN_NAME:
        print(f"Found '{EXPECTED_COLUMNS_IN_FILE[-1]}' column. Renaming it to '{TARGET_COLUMN_NAME}' for internal use.")
        df.rename(columns={EXPECTED_COLUMNS_IN_FILE[-1]: TARGET_COLUMN_NAME}, inplace=True)
    # If after all checks, the internal TARGET_COLUMN_NAME is not in df, something is wrong.
    elif TARGET_COLUMN_NAME not in df.columns:
        raise ValueError(f"Could not find or establish the target column '{TARGET_COLUMN_NAME}'. Please check CSV column names.")

    print("\nDataFrame columns after potential rename:", df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found. Please make sure it's in the same directory as the script or provide the full path.")
    exit()
except ValueError as e:
    print(f"Error during column validation: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the CSV: {e}")
    exit()

# --- Data Cleaning ---
print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

# Define columns critical for model input and basic info; common names are for info, not direct model input features.
# However, converting them to string early helps avoid issues.
cols_to_convert_to_str = [
    'Plant_A_Name', 'A_Family', 'A_Genus', 'A_Common_Name',
    'Plant_B_Name', 'B_Family', 'B_Genus', 'B_Common_Name'
]
for col in cols_to_convert_to_str:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('<UNK>') # Fill NaNs in string columns with '<UNK>'

# For Family/Genus specifically (model features), ensure NaNs are '<UNK>' before creating maps
# This ensures '<UNK>' is a known category if present as NaN
# This step might be redundant if the loop above already handled it, but it's a safeguard.
for col in ['A_Family', 'A_Genus', 'B_Family', 'B_Genus']:
    if col in df.columns:
        df[col].fillna('<UNK>', inplace=True) # Ensure string type first if not done already.

# Drop rows where the target (Compatibility) is NaN
df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)
print(f"\nDataFrame shape after dropping rows with NaN in '{TARGET_COLUMN_NAME}': {df.shape}")

# Ensure Compatibility is numeric; coerce errors will turn non-numeric to NaN
df[TARGET_COLUMN_NAME] = pd.to_numeric(df[TARGET_COLUMN_NAME], errors='coerce')
df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True) # Drop rows again if coercion created NaNs
print(f"\nDataFrame shape after ensuring '{TARGET_COLUMN_NAME}' is numeric and dropping resulting NaNs: {df.shape}")

if df.empty:
    print("Error: DataFrame is empty after loading or cleaning. Cannot proceed.")
    exit()

# Create the plant information lookup table from the loaded data
# This table will include common names and be saved for the prediction part.
plant_a_info = df[['Plant_A_Name', 'A_Family', 'A_Genus', 'A_Common_Name']].rename(
    columns={'Plant_A_Name': 'Name', 'A_Family': 'Family', 'A_Genus': 'Genus', 'A_Common_Name': 'Common_Name'}
)
plant_b_info = df[['Plant_B_Name', 'B_Family', 'B_Genus', 'B_Common_Name']].rename(
    columns={'Plant_B_Name': 'Name', 'B_Family': 'Family', 'B_Genus': 'Genus', 'B_Common_Name': 'Common_Name'}
)
# Concatenate and remove duplicates. drop_duplicates keeps the first occurrence.
plant_info_df = pd.concat([plant_a_info, plant_b_info]).drop_duplicates(subset=['Name'])
# Set 'Name' (scientific name) as the index for easy lookup
plant_info_df = plant_info_df.set_index('Name')
# Fill any remaining NaN common names with 'N/A' for display purposes
plant_info_df['Common_Name'].fillna('N/A', inplace=True)
# Family and Genus columns in plant_info_df will have '<UNK>' if original was NaN and processed above.

print("\nPlant Info Lookup Table (sample from plant_info_df):")
print(plant_info_df.head())


# Create integer encodings for categorical features used by the model (Name, Family, Genus)
# Common names are NOT encoded as they are not direct model inputs.
all_plant_names = pd.concat([df['Plant_A_Name'], df['Plant_B_Name']]).astype(str).unique()
all_families = pd.concat([df['A_Family'], df['B_Family']]).astype(str).unique() # NaNs should be '<UNK>'
all_genera = pd.concat([df['A_Genus'], df['B_Genus']]).astype(str).unique()   # NaNs should be '<UNK>'

plant_name_map = {name: i for i, name in enumerate(all_plant_names)}
plant_name_map['<UNK>'] = len(plant_name_map) # Add <UNK> token for plant names not seen in training

family_map = {name: i for i, name in enumerate(all_families)}
if '<UNK>' not in family_map: # Ensure <UNK> is in map, critical if all families were known
    family_map['<UNK>'] = len(family_map)

genus_map = {name: i for i, name in enumerate(all_genera)}
if '<UNK>' not in genus_map: # Ensure <UNK> is in map
    genus_map['<UNK>'] = len(genus_map)

# Inverse maps (integer to name) can be useful for debugging but not strictly needed for prediction logic here
# inv_plant_name_map = {v: k for k, v in plant_name_map.items()}
# inv_family_map = {v: k for k, v in family_map.items()}
# inv_genus_map = {v: k for k, v in genus_map.items()}

vocab_size_name = len(plant_name_map)
vocab_size_family = len(family_map)
vocab_size_genus = len(genus_map)

print(f"\nVocab sizes: Name={vocab_size_name}, Family={vocab_size_family}, Genus={vocab_size_genus}")

# Apply encoding to the DataFrame features for model training
df_encoded = df.copy()
# Use .get with default to plant_name_map['<UNK>'] for robustness
df_encoded['Plant_A_Name_Enc'] = df_encoded['Plant_A_Name'].astype(str).map(lambda x: plant_name_map.get(x, plant_name_map['<UNK>']))
df_encoded['A_Family_Enc'] = df_encoded['A_Family'].astype(str).map(lambda x: family_map.get(x, family_map['<UNK>']))
df_encoded['A_Genus_Enc'] = df_encoded['A_Genus'].astype(str).map(lambda x: genus_map.get(x, genus_map['<UNK>']))
df_encoded['Plant_B_Name_Enc'] = df_encoded['Plant_B_Name'].astype(str).map(lambda x: plant_name_map.get(x, plant_name_map['<UNK>']))
df_encoded['B_Family_Enc'] = df_encoded['B_Family'].astype(str).map(lambda x: family_map.get(x, family_map['<UNK>']))
df_encoded['B_Genus_Enc'] = df_encoded['B_Genus'].astype(str).map(lambda x: genus_map.get(x, genus_map['<UNK>']))

print("\nEncoded DataFrame (sample):")
print(df_encoded[['Plant_A_Name_Enc', 'A_Family_Enc', 'A_Genus_Enc', 'Plant_B_Name_Enc', 'B_Family_Enc', 'B_Genus_Enc', TARGET_COLUMN_NAME]].head())

# Prepare model inputs (X) and target (y)
X = {
    'plant_a_name': np.array(df_encoded['Plant_A_Name_Enc']),
    'a_family': np.array(df_encoded['A_Family_Enc']),
    'a_genus': np.array(df_encoded['A_Genus_Enc']),
    'plant_b_name': np.array(df_encoded['Plant_B_Name_Enc']),
    'b_family': np.array(df_encoded['B_Family_Enc']),
    'b_genus': np.array(df_encoded['B_Genus_Enc'])
}
y = np.array(df_encoded[TARGET_COLUMN_NAME]) # Compatibility scores (0-100)

# Split data into training and testing sets
if len(y) < 10:
    print("\nWarning: Dataset is very small. Splitting into train/test may not be effective.")
    print("Consider getting more data. For now, using all data for training if size < 10, with a small validation split during fit if possible.")
    if len(y) < 2:
        print("Error: Not enough data to even form a single batch. Need at least 2 samples.")
        exit()

X_train_dict, X_test_dict, y_train, y_test = {}, {}, np.array([]), np.array([])
test_size_ratio = 0.2 if len(y) >= 50 else (0.1 if len(y) >= 10 else 0) # Dynamic test size

if test_size_ratio == 0 and len(y) > 0:
    print("Using all data for training, validation split will be attempted during fit if data is sufficient.")
    X_train_dict = X
    y_train = y
    X_test_dict = {key: np.array([]) for key in X.keys()} # Empty test set
    y_test = np.array([])
elif len(y) > 0 :
    try:
        # Stratify if target variable has enough classes, otherwise split normally
        stratify_option = y if len(np.unique(y)) >= 2 else None
        train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=test_size_ratio, random_state=42, stratify=stratify_option)
        for key_X in X:
            X_train_dict[key_X] = X[key_X][train_indices]
            X_test_dict[key_X] = X[key_X][test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
    except ValueError as e:
        print(f"Warning: Could not stratify train/test split (Error: {e}). Splitting without stratification.")
        train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=test_size_ratio, random_state=42)
        for key_X in X:
            X_train_dict[key_X] = X[key_X][train_indices]
            X_test_dict[key_X] = X[key_X][test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
else: # len(y) == 0
    print("Error: No data available after pre-processing. Cannot proceed with train/test split.")
    exit()

print(f"\nTraining samples: {len(y_train)}, Test samples: {len(y_test)}")
if len(y_train) == 0 :
    print("Error: Training set is empty after split. Check data size and split logic.")
    exit()
if test_size_ratio > 0 and len(y_test) == 0:
     print("Warning: Test set is empty after split, though a test_size_ratio > 0 was specified. This might happen with very small datasets.")

# --- 2. Model Building ---
# Dynamically adjust embedding dimensions based on vocabulary size
embedding_dim_name = max(2, min(50, vocab_size_name // 4 if vocab_size_name > 3 else 2))
embedding_dim_family = max(2, min(20, vocab_size_family // 2 if vocab_size_family > 3 else 2))
embedding_dim_genus = max(2, min(30, vocab_size_genus // 2 if vocab_size_genus > 3 else 2))

print(f"Embedding Dims: Name={embedding_dim_name}, Family={embedding_dim_family}, Genus={embedding_dim_genus}")

# Define Input layers for each feature
input_plant_a_name = layers.Input(shape=(1,), name='plant_a_name', dtype='int32')
input_a_family = layers.Input(shape=(1,), name='a_family', dtype='int32')
input_a_genus = layers.Input(shape=(1,), name='a_genus', dtype='int32')
input_plant_b_name = layers.Input(shape=(1,), name='plant_b_name', dtype='int32')
input_b_family = layers.Input(shape=(1,), name='b_family', dtype='int32')
input_b_genus = layers.Input(shape=(1,), name='b_genus', dtype='int32')

# Embedding layers for each categorical feature
emb_plant_a_name = layers.Embedding(input_dim=vocab_size_name, output_dim=embedding_dim_name)(input_plant_a_name)
emb_a_family = layers.Embedding(input_dim=vocab_size_family, output_dim=embedding_dim_family)(input_a_family)
emb_a_genus = layers.Embedding(input_dim=vocab_size_genus, output_dim=embedding_dim_genus)(input_a_genus)
emb_plant_b_name = layers.Embedding(input_dim=vocab_size_name, output_dim=embedding_dim_name)(input_plant_b_name)
emb_b_family = layers.Embedding(input_dim=vocab_size_family, output_dim=embedding_dim_family)(input_b_family)
emb_b_genus = layers.Embedding(input_dim=vocab_size_genus, output_dim=embedding_dim_genus)(input_b_genus)

# Flatten the embedding outputs
flat_plant_a_name = layers.Flatten()(emb_plant_a_name)
flat_a_family = layers.Flatten()(emb_a_family)
flat_a_genus = layers.Flatten()(emb_a_genus)
flat_plant_b_name = layers.Flatten()(emb_plant_b_name)
flat_b_family = layers.Flatten()(emb_b_family)
flat_b_genus = layers.Flatten()(emb_b_genus)

# Concatenate all flattened features
concatenated = layers.Concatenate()([
    flat_plant_a_name, flat_a_family, flat_a_genus,
    flat_plant_b_name, flat_b_family, flat_b_genus
])

# Dense layers with dropout, size adapted to dataset size
if len(y_train) > 10000:
    dense_units1 = 128
    dense_units2 = 64
    dropout_rate = 0.4
elif len(y_train) > 1000:
    dense_units1 = 64
    dense_units2 = 32
    dropout_rate = 0.3
else: # Smaller datasets
    total_embedding_dim = embedding_dim_name * 2 + embedding_dim_family * 2 + embedding_dim_genus * 2
    dense_units1 = max(16, total_embedding_dim // 2 if total_embedding_dim > 0 else 16)
    dense_units2 = max(8, dense_units1 // 2)
    dropout_rate = 0.2

dense1 = layers.Dense(dense_units1, activation='relu')(concatenated)
dropout1 = layers.Dropout(dropout_rate)(dense1)

if len(y_train) > 50 and dense_units1 > dense_units2 : # Add second dense layer if dataset is large enough
    dense2 = layers.Dense(dense_units2, activation='relu')(dropout1)
    dropout2 = layers.Dropout(dropout_rate)(dense2)
    final_dense_input = dropout2
else:
    final_dense_input = dropout1

# Output layer: predicts a single value (compatibility score)
# Linear activation as we are predicting a continuous value (0-100)
output = layers.Dense(1, activation='linear', name='compatibility_output')(final_dense_input)

model = keras.Model(
    inputs=[input_plant_a_name, input_a_family, input_a_genus,
            input_plant_b_name, input_b_family, input_b_genus],
    outputs=output
)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', # Good for regression
              metrics=['mean_absolute_error']) # MAE is interpretable (avg error in % points)

model.summary()

# --- 3. Model Training ---
epochs = 50 # Number of times to iterate over the entire training dataset
# Adjust batch size based on training set size
base_batch_size = 256 if len(y_train) > 5000 else (64 if len(y_train) > 1000 else 32)
batch_size = min(base_batch_size, len(y_train)) if len(y_train) > 0 else 1
if batch_size == 0 and len(y_train) > 0: batch_size = 1 # Ensure batch_size is at least 1

# Early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=10,             # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restores model weights from the epoch with the best value of the monitored quantity.
    verbose=1
)

history = None # Initialize history object
callbacks_for_fit = []
validation_args_for_fit = {}

# Determine if validation_split can be used from the training data
# Requires enough samples for both effective training and a meaningful validation set.
min_val_samples = max(5, batch_size // 4 if batch_size > 0 else 5)
if batch_size > 0 and (len(y_train) * 0.2 >= min_val_samples) and (len(y_train) * 0.8 >= batch_size):
    validation_args_for_fit['validation_split'] = 0.2
    callbacks_for_fit.append(early_stopping)
    print(f"Using validation_split=0.2. Effective training samples: {int(len(y_train)*0.8)}, validation samples: {int(len(y_train)*0.2)}")
else:
    print("Warning: Not enough data for a meaningful validation_split from the training set, or remaining training data too small.")
    print("Training without validation_split. EarlyStopping on 'val_loss' will not be active unless a dedicated validation set is provided (not implemented here).")


print(f"Starting training: epochs={epochs}, batch_size={batch_size}")
if len(y_train) > 0 and batch_size > 0 :
    history = model.fit(
        X_train_dict,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_for_fit,
        verbose=1,
        **validation_args_for_fit # This will pass validation_split if it was set
    )
else:
    print("Skipping training as there is no training data or batch size is zero.")


# --- 4. Save Mappings, Plant Info, and Model ---
print("\n--- Saving Artifacts ---")
# Save plant_info_df (the lookup table with common names)
try:
    plant_info_df.to_csv(PLANT_INFO_SAVE_PATH)
    print(f"Plant info lookup table saved to {PLANT_INFO_SAVE_PATH}")
except Exception as e:
    print(f"Error saving plant_info_df: {e}")

# Save mapping dictionaries (for converting names/families/genera to integers)
try:
    with open(PLANT_NAME_MAP_SAVE_PATH, 'wb') as f: pickle.dump(plant_name_map, f)
    print(f"Plant name map saved to {PLANT_NAME_MAP_SAVE_PATH}")
    with open(FAMILY_MAP_SAVE_PATH, 'wb') as f: pickle.dump(family_map, f)
    print(f"Family map saved to {FAMILY_MAP_SAVE_PATH}")
    with open(GENUS_MAP_SAVE_PATH, 'wb') as f: pickle.dump(genus_map, f)
    print(f"Genus map saved to {GENUS_MAP_SAVE_PATH}")
except Exception as e:
    print(f"Error saving mapping dictionaries: {e}")

# Save the Trained Model
if history and y_train.size > 0: # Ensure model was trained
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
else:
    print("\nSkipping model saving as training might not have occurred or was on an empty/insufficient dataset.")

# --- 5. Model Evaluation ---
if history and len(y_test) > 0 and X_test_dict['plant_a_name'].size > 0 :
    print("\n--- Evaluating on Test Set ---")
    eval_batch_size = min(batch_size, len(y_test)) if len(y_test) > 0 else 1
    if eval_batch_size == 0 and len(y_test) > 0 : eval_batch_size = 1

    if eval_batch_size > 0:
        loss, mae = model.evaluate(X_test_dict, y_test, verbose=0, batch_size=eval_batch_size)
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test Mean Absolute Error (MAE): {mae:.4f} (compatibility % points)")
    else:
        print("Test set is empty or evaluation batch size is zero. Skipping evaluation.")

elif test_size_ratio == 0 and history and len(y_train) > 0: # If all data was used for training
    print("\n--- Evaluating on Training Set (as no dedicated test set was split) ---")
    eval_batch_size = min(batch_size, len(y_train)) if len(y_train) > 0 else 1
    if eval_batch_size == 0 and len(y_train) > 0 : eval_batch_size = 1

    if eval_batch_size > 0:
        loss, mae = model.evaluate(X_train_dict, y_train, verbose=0, batch_size=eval_batch_size)
        print(f"Train Loss (MSE): {loss:.4f}") # This is on the full training set
        print(f"Train Mean Absolute Error (MAE): {mae:.4f} (compatibility % points)")
    else:
         print("Training set is empty or evaluation batch size is zero. Skipping evaluation.")
else:
    print("\nSkipping test set evaluation due to insufficient/no test data or no training occurred.")


# --- 6. Prediction Functions ---

def predict_single_compatibility(plant_a_sci_name, plant_b_sci_name, trained_model,
                                 current_plant_info_df, current_plant_name_map,
                                 current_family_map, current_genus_map):
    """
    Predicts compatibility between two specific plants.
    This is a helper for the find_graftable_plants function.
    """
    plant_a_sci_name = str(plant_a_sci_name)
    plant_b_sci_name = str(plant_b_sci_name)

    # Get Family and Genus for Plant A
    try:
        info_a = current_plant_info_df.loc[plant_a_sci_name]
        family_a = str(info_a['Family'])
        genus_a = str(info_a['Genus'])
    except KeyError: # Plant A not in our lookup table
        family_a = '<UNK>'
        genus_a = '<UNK>'

    # Get Family and Genus for Plant B
    try:
        info_b = current_plant_info_df.loc[plant_b_sci_name]
        family_b = str(info_b['Family'])
        genus_b = str(info_b['Genus'])
    except KeyError: # Plant B not in our lookup table
        family_b = '<UNK>'
        genus_b = '<UNK>'

    # Prepare input data for the model
    # Use .get(key, map['<UNK>']) to handle items not seen in training (they'll map to <UNK> token)
    input_data = {
        'plant_a_name': np.array([current_plant_name_map.get(plant_a_sci_name, current_plant_name_map['<UNK>'])]),
        'a_family': np.array([current_family_map.get(family_a, current_family_map['<UNK>'])]),
        'a_genus': np.array([current_genus_map.get(genus_a, current_genus_map['<UNK>'])]),
        'plant_b_name': np.array([current_plant_name_map.get(plant_b_sci_name, current_plant_name_map['<UNK>'])]),
        'b_family': np.array([current_family_map.get(family_b, current_family_map['<UNK>'])]),
        'b_genus': np.array([current_genus_map.get(genus_b, current_genus_map['<UNK>'])])
    }

    prediction = trained_model.predict(input_data, verbose=0)
    return max(0, min(100, prediction[0][0])) # Clip prediction to 0-100 range

def find_graftable_plants(target_plant_sci_name, trained_model, current_plant_info_df,
                          current_plant_name_map, current_family_map, current_genus_map,
                          compatibility_threshold=60):
    """
    Finds plants graftable with the target_plant_sci_name.
    Returns a list of tuples: (scientific_name, common_name, predicted_compatibility_score)
    for plants exceeding the compatibility_threshold.
    """
    target_plant_sci_name = str(target_plant_sci_name)
    if target_plant_sci_name not in current_plant_info_df.index:
        print(f"Warning: Target plant '{target_plant_sci_name}' not found in the plant lookup table. Cannot provide its family/genus accurately.")
        # We can still proceed, it will be treated as '<UNK>' for its own features if not in name_map

    compatible_plants = []
    print(f"\nSearching for plants compatible with '{target_plant_sci_name}' (Threshold: >{compatibility_threshold}%)...")

    # Iterate through all unique plants in our lookup table (plant_info_df)
    for potential_partner_sci_name in current_plant_info_df.index:
        if potential_partner_sci_name == target_plant_sci_name:
            continue # Don't compare a plant with itself for "other" graftable plants

        # The predict_single_compatibility function will handle lookups and <UNK> for both plants
        predicted_score = predict_single_compatibility(
            target_plant_sci_name,
            potential_partner_sci_name,
            trained_model,
            current_plant_info_df,
            current_plant_name_map,
            current_family_map,
            current_genus_map
        )

        if predicted_score > compatibility_threshold:
            partner_common_name = current_plant_info_df.loc[potential_partner_sci_name, 'Common_Name']
            compatible_plants.append({
                "Scientific Name": potential_partner_sci_name,
                "Common Name": partner_common_name,
                "Predicted Compatibility (%)": round(predicted_score, 2)
            })

    # Sort by compatibility score in descending order
    compatible_plants.sort(key=lambda x: x["Predicted Compatibility (%)"], reverse=True)
    return compatible_plants


# --- 7. Example Predictions & Usage ---
print("\n--- Example Usage (using the trained model from this session) ---")

# Check if model training was successful and data is available
model_ready = 'model' in locals() and model is not None and history is not None and not plant_info_df.empty
if model_ready:
    # Example 1: Pairwise prediction (similar to original script's example)
    if len(df_encoded) >=1 :
        sample_row = df_encoded.sample(n=1, replace=False) # Use df_encoded to get original names from a sample
        plant_a_ex = sample_row.iloc[0]['Plant_A_Name']
        plant_b_ex = sample_row.iloc[0]['Plant_B_Name']
        actual_ex = sample_row.iloc[0][TARGET_COLUMN_NAME]

        pred_ex = predict_single_compatibility(plant_a_ex, plant_b_ex, model, plant_info_df,
                                            plant_name_map, family_map, genus_map)
        print(f"\nPairwise Prediction Example:")
        print(f"  {plant_a_ex} + {plant_b_ex}: Predicted={pred_ex:.2f}%, Actual (from dataset)={actual_ex:.2f}%")

    # Example 2: Finding graftable plants for a specific plant
    # Pick a plant from your dataset to test with, e.g., the first Plant_A_Name
    if not df.empty:
        test_plant_name = df['Plant_A_Name'].iloc[0]
        print(f"\nFinding graftable companions for: {test_plant_name}")

        # You can adjust the threshold
        graftable_list = find_graftable_plants(test_plant_name, model, plant_info_df,
                                               plant_name_map, family_map, genus_map,
                                               compatibility_threshold=65) # Example threshold

        if graftable_list:
            print(f"\nFound {len(graftable_list)} potential graftable plants for '{test_plant_name}':")
            for plant_info in graftable_list:
                print(f"  - Scientific: {plant_info['Scientific Name']}, Common: {plant_info['Common Name']}, Score: {plant_info['Predicted Compatibility (%)']}%")
        else:
            print(f"No plants found exceeding the compatibility threshold for '{test_plant_name}'. Try a lower threshold.")

        # Example with a plant name that might be completely unknown (tests <UNK> handling)
        unknown_plant_test = "Unknownus plantus_XYZ"
        print(f"\nFinding graftable companions for a potentially unknown plant: {unknown_plant_test}")
        graftable_unknown_list = find_graftable_plants(unknown_plant_test, model, plant_info_df,
                                               plant_name_map, family_map, genus_map,
                                               compatibility_threshold=50)
        if graftable_unknown_list:
            print(f"\nFound {len(graftable_unknown_list)} potential graftable plants for '{unknown_plant_test}':")
            for plant_info in graftable_unknown_list:
                 print(f"  - Scientific: {plant_info['Scientific Name']}, Common: {plant_info['Common Name']}, Score: {plant_info['Predicted Compatibility (%)']}%")
        else:
            print(f"No plants found exceeding the compatibility threshold for '{unknown_plant_test}'.")

    else:
        print("Original DataFrame 'df' is empty, cannot pick a test plant name.")

else:
    print("Model not trained or data not available; cannot generate example predictions.")


# --- Optional: Plot training history ---
if history and hasattr(history, 'history') and history.history:
    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    if 'val_loss' in history.history: # Check if validation loss exists
        plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot Training & Validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    if 'val_mean_absolute_error' in history.history: # Check if validation MAE exists
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (Compatibility % points)')
    plt.legend()

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()
else:
    print("\nNo training history to plot (e.g., training skipped, failed, or no validation data).")

print("\n--- Graft Compatibility Training and Prediction Script Finished ---")