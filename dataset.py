import csv

# --- Configuration ---
INPUT_CSV_FILE = 'clean.csv'
OUTPUT_CSV_FILE = 'plant_compatibility.csv'

# Compatibility scores
SCORE_SAME_SPECIES = 100 # If Plant A Name is exactly Plant B Name
SCORE_SAME_GENUS = 95
SCORE_SAME_FAMILY = 65
SCORE_DIFFERENT_FAMILY = 15

# --- Main Logic ---
def calculate_compatibility():
    plants = []
    # 1. Read the input CSV
    try:
        with open(INPUT_CSV_FILE, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames or not all(f in reader.fieldnames for f in ['Name', 'Family', 'Genus', 'Common_Name']):
                print(f"Error: Input CSV must contain 'Name', 'Family', 'Genus', 'Common_Name' columns.")
                return
            for row in reader:
                plants.append(row)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    if not plants:
        print("No data found in input CSV.")
        return

    output_data = []
    # Define header for the output CSV
    header = [
        'Plant_A_Name', 'A_Family', 'A_Genus', 'A_Common_Name',
        'Plant_B_Name', 'B_Family', 'B_Genus', 'B_Common_Name',
        'Compatibility (%)'
    ]
    output_data.append(header)

    # 2. Compare each plant with every other plant (including itself)
    for plant_a in plants:
        for plant_b in plants:
            compatibility_score = 0

            # Check for same species (exact name match) first
            if plant_a['Name'] == plant_b['Name']:
                compatibility_score = SCORE_SAME_SPECIES
            # Then check for same genus
            elif plant_a['Genus'] == plant_b['Genus']:
                compatibility_score = SCORE_SAME_GENUS
            # Then check for same family
            elif plant_a['Family'] == plant_b['Family']:
                compatibility_score = SCORE_SAME_FAMILY
            # Otherwise, they are in different families
            else:
                compatibility_score = SCORE_DIFFERENT_FAMILY

            output_row = [
                plant_a['Name'], plant_a['Family'], plant_a['Genus'], plant_a['Common_Name'],
                plant_b['Name'], plant_b['Family'], plant_b['Genus'], plant_b['Common_Name'],
                compatibility_score
            ]
            output_data.append(output_row)

    # 3. Write the output CSV
    try:
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(output_data)
        print(f"Successfully generated '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Error writing output CSV: {e}")

# --- Run the script ---
if __name__ == "__main__":
    # You can create a dummy plants_data.csv for testing if it doesn't exist
    # For example:
    # with open(INPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Name','Family','Genus','Common_Name'])
    #     writer.writerow(['Helianthus annuus','Asteraceae','Helianthus','Common Sunflower'])
    #     writer.writerow(['Helianthus tuberosus','Asteraceae','Helianthus','Jerusalem Artichoke'])
    #     writer.writerow(['Taraxacum officinale','Asteraceae','Taraxacum','Common Dandelion'])
    #     writer.writerow(['Rosa gallica','Rosaceae','Rosa','Gallica Rose'])

    calculate_compatibility()