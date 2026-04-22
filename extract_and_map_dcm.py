

import os
import shutil
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

# Configuration
ADNI_FOLDER = "ADNI"
CSV_FILE = "ADNIMERGE.csv"
OUTPUT_FOLDER = "ADNI_Mapped"

def extract_ptid_from_path(dcm_path):
    """
    Extract Patient ID (PTID) from DCM file path or filename
    Example: ADNI/023_S_1289/... -> 023_S_1289
    """
    parts = Path(dcm_path).parts
    for part in parts:
        # Match pattern like 023_S_1289
        if re.match(r'\d{3}_S_\d{4}', part):
            return part
    return None

def extract_ptid_from_filename(filename):
    """
    Extract Patient ID from DCM filename
    Example: ADNI_023_S_1289_MR_MPRAGE_... -> 023_S_1289
    """
    match = re.search(r'(\d{3}_S_\d{4})', filename)
    if match:
        return match.group(1)
    return None

def find_all_dcm_files(adni_folder):
    """
    Recursively find all .dcm files in ADNI folder
    """
    dcm_files = []
    print(f"Searching for DCM files in {adni_folder}...")

    for root, dirs, files in os.walk(adni_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                full_path = os.path.join(root, file)
                dcm_files.append(full_path)

    print(f"Found {len(dcm_files)} DCM files")
    return dcm_files

def load_adnimerge_data(csv_file):
    """
    Load ADNIMERGE CSV and get unique patient information
    """
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    # Get baseline (bl) records for each patient
    df_baseline = df[df['VISCODE'] == 'bl'].copy()

    print(f"Loaded {len(df)} total records")
    print(f"Found {len(df_baseline)} baseline records")
    print(f"Unique patients: {df['PTID'].nunique()}")

    return df, df_baseline

def create_mapped_structure(dcm_files, df_baseline, output_folder):
    """
    Create organized folder structure with mapped DCM files
    """
    os.makedirs(output_folder, exist_ok=True)

    # Create a mapping summary
    mapping_summary = []
    patients_with_files = set()
    patients_without_csv = set()

    print(f"\nProcessing {len(dcm_files)} DCM files...")

    for dcm_path in tqdm(dcm_files, desc="Copying files"):
        # Extract PTID from path
        ptid = extract_ptid_from_path(dcm_path)
        if not ptid:
            # Try extracting from filename
            ptid = extract_ptid_from_filename(os.path.basename(dcm_path))

        if not ptid:
            print(f"Warning: Could not extract PTID from {dcm_path}")
            continue

        patients_with_files.add(ptid)

        # Check if PTID exists in CSV
        patient_data = df_baseline[df_baseline['PTID'] == ptid]

        if patient_data.empty:
            patients_without_csv.add(ptid)
            diagnosis = "UNKNOWN"
        else:
            diagnosis = patient_data.iloc[0]['DX_bl']
            if pd.isna(diagnosis):
                diagnosis = "UNKNOWN"

        # Create output directory structure: OUTPUT_FOLDER/DIAGNOSIS/PTID/
        output_dir = os.path.join(output_folder, str(diagnosis), ptid)
        os.makedirs(output_dir, exist_ok=True)

        # Copy file to new location
        filename = os.path.basename(dcm_path)
        output_path = os.path.join(output_dir, filename)

        # Copy file (preserve original)
        shutil.copy2(dcm_path, output_path)

        # Add to mapping summary
        mapping_summary.append({
            'PTID': ptid,
            'Original_Path': dcm_path,
            'New_Path': output_path,
            'Diagnosis': diagnosis
        })

    return mapping_summary, patients_with_files, patients_without_csv

def save_mapping_report(mapping_summary, patients_with_files, patients_without_csv,
                        df_baseline, output_folder):
    """
    Save detailed mapping report
    """
    # Save mapping summary to CSV
    summary_df = pd.DataFrame(mapping_summary)
    summary_path = os.path.join(output_folder, 'mapping_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMapping summary saved to: {summary_path}")

    # Create statistics report
    stats_path = os.path.join(output_folder, 'mapping_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ADNI DCM FILES MAPPING STATISTICS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total DCM files processed: {len(mapping_summary)}\n")
        f.write(f"Unique patients with DCM files: {len(patients_with_files)}\n")
        f.write(f"Patients with DCM but not in CSV: {len(patients_without_csv)}\n\n")

        # Diagnosis distribution
        f.write("Distribution by Diagnosis:\n")
        f.write("-" * 40 + "\n")
        diagnosis_counts = summary_df.groupby('Diagnosis').size().sort_values(ascending=False)
        for diagnosis, count in diagnosis_counts.items():
            f.write(f"{diagnosis}: {count} files\n")

        f.write("\n")
        f.write(f"Unique patients by diagnosis:\n")
        f.write("-" * 40 + "\n")
        patient_diagnosis_counts = summary_df.groupby('Diagnosis')['PTID'].nunique().sort_values(ascending=False)
        for diagnosis, count in patient_diagnosis_counts.items():
            f.write(f"{diagnosis}: {count} patients\n")

        if patients_without_csv:
            f.write("\n")
            f.write("Patients with DCM files but not in CSV baseline:\n")
            f.write("-" * 40 + "\n")
            for ptid in sorted(patients_without_csv):
                f.write(f"{ptid}\n")

    print(f"Statistics report saved to: {stats_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total DCM files copied: {len(mapping_summary)}")
    print(f"Unique patients: {len(patients_with_files)}")
    print(f"\nFiles organized by diagnosis in: {output_folder}")
    print("\nDiagnosis distribution:")
    for diagnosis, count in patient_diagnosis_counts.items():
        print(f"  {diagnosis}: {count} patients")

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("ADNI DCM Extraction and Mapping Tool")
    print("=" * 60)
    print()

    # Check if required files/folders exist
    if not os.path.exists(ADNI_FOLDER):
        print(f"Error: ADNI folder not found: {ADNI_FOLDER}")
        return

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found: {CSV_FILE}")
        return

    # Step 1: Find all DCM files
    dcm_files = find_all_dcm_files(ADNI_FOLDER)
    if not dcm_files:
        print("No DCM files found!")
        return

    # Step 2: Load ADNIMERGE data
    df, df_baseline = load_adnimerge_data(CSV_FILE)

    # Step 3: Create mapped structure
    mapping_summary, patients_with_files, patients_without_csv = create_mapped_structure(
        dcm_files, df_baseline, OUTPUT_FOLDER
    )

    # Step 4: Save reports
    save_mapping_report(mapping_summary, patients_with_files, patients_without_csv,
                       df_baseline, OUTPUT_FOLDER)

    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
