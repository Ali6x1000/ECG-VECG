import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# --- Configuration ---
RESULTS_DIR = "./checkpoints/feb-27"  
DATA_DIR = "./data" # Ensure patients.csv.gz and admissions.csv.gz are here
CACHE_DIR = "/scratch/pioneer/users/aan90/mimic_cache" 

# 1. Load and Prepare Clinical Metadata
def load_clinical_data():
    print("Loading Clinical Data...")
    
    # Load Patients (Outcomes)
    df_pat = pd.read_csv(os.path.join(DATA_DIR, "patients.csv.gz"))
    
    # Load Admissions (Censoring Time)
    df_adm = pd.read_csv(os.path.join(DATA_DIR, "admissions.csv.gz"))
    df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])
    
    # Find the LATEST contact date for each patient
    last_contact = df_adm.sort_values('dischtime').groupby('subject_id')['dischtime'].last().reset_index()
    last_contact.rename(columns={'dischtime': 'last_contact_date'}, inplace=True)
    
    # Merge
    df = pd.merge(df_pat, last_contact, on='subject_id', how='left')
    
    # Define Event (Dead=1, Alive=0)
    df['event'] = df['dod'].notnull().astype(int)
    
    # FIX 1: Return the raw dates so we can calculate exact duration later
    return df[['subject_id', 'event', 'gender', 'dod', 'last_contact_date']]

# 2. Run Survival Analysis
def run_analysis(model_name):
    json_path = os.path.join(RESULTS_DIR, f"{model_name}_test_results.json")
    if not os.path.exists(json_path): 
        print(f"⚠️ Skipping {model_name}: File not found at {json_path}")
        return

    with open(json_path, 'r') as f: 
        data = json.load(f)

    meta_path = os.path.join(CACHE_DIR, f"{model_name}_mimic_metadata.csv")
    cache_meta = pd.read_csv(meta_path)
    if 'fold' not in cache_meta: cache_meta['fold'] = cache_meta['subject_id'] % 10
    
    # Isolate test fold
    test_meta = cache_meta[cache_meta['fold'] == 9].copy()
    
    # 1. CALCULATE EXACT AGE & DURATION START TIME
    test_meta['ecg_time'] = pd.to_datetime(test_meta['ecg_time'])
    test_meta['ecg_year'] = test_meta['ecg_time'].dt.year
    test_meta['true_age_at_ecg'] = test_meta['anchor_age'] + (test_meta['ecg_year'] - test_meta['anchor_year'])
    
    # FIX 2: ALIGNMENT SAFETY - Drop artifacts so rows match JSON exactly
    test_meta = test_meta[(test_meta['true_age_at_ecg'] >= 18) & (test_meta['true_age_at_ecg'] <= 90)].copy()
    
    # Clip to match JSON predictions (Just in case)
    limit = min(len(test_meta), len(data['predictions']))
    test_meta = test_meta.iloc[:limit].copy()
    
    # Add predictions
    test_meta['pred_age'] = data['predictions'][:limit]
    test_meta['delta_age'] = test_meta['pred_age'] - test_meta['true_age_at_ecg']
    
    # 2. GROUP BY PATIENT
    patient_preds = test_meta.groupby('subject_id').agg({
        'delta_age': 'mean',
        'true_age_at_ecg': 'mean',
        'ecg_time': 'min' # Start the survival clock at their FIRST ECG
    }).reset_index()

    # 3. MERGE WITH OUTCOMES
    df_outcomes = load_clinical_data()
    df_final = pd.merge(patient_preds, df_outcomes, on='subject_id', how='inner')
    
    # 4. FIX DURATION (Time from First ECG to Death/Censoring)
    df_final['dod'] = pd.to_datetime(df_final['dod'], errors='coerce')
    df_final['last_contact_date'] = pd.to_datetime(df_final['last_contact_date'], errors='coerce')
    
    # Duration in Days, converted to Years
    mask_dead = df_final['event'] == 1
    df_final.loc[mask_dead, 'duration'] = (df_final.loc[mask_dead, 'dod'] - df_final.loc[mask_dead, 'ecg_time']).dt.days / 365.25
    df_final.loc[~mask_dead, 'duration'] = (df_final.loc[~mask_dead, 'last_contact_date'] - df_final.loc[~mask_dead, 'ecg_time']).dt.days / 365.25
    
    # Clean up edge cases
    df_final.loc[df_final['duration'] <= 0, 'duration'] = 0.01 
    

    # Normalize Delta for the Cox Model
    df_final['delta_z'] = (df_final['delta_age'] - df_final['delta_age'].mean()) / df_final['delta_age'].std()
    df_final['is_male'] = (df_final['gender'] == 'M').astype(int)
    
    # 🚨 ADD THIS EXACT LINE HERE 🚨
    df_final = df_final.dropna(subset=['duration', 'event', 'delta_age', 'true_age_at_ecg', 'is_male']).copy()
    
    cols = ['duration', 'event', 'delta_z', 'true_age_at_ecg', 'is_male']
    data_for_cph = df_final[cols] # Removed the .dropna() here since we did it above
    
    # --- COX MODEL FITTING ---
    cph = CoxPHFitter()
    cph.fit(data_for_cph, duration_col='duration', event_col='event')
    print(f"\n📊 Corrected Analysis for {model_name} (Unique Patients={len(data_for_cph)})")
    cph.print_summary()

    # --- KAPLAN-MEIER PLOTTING ---
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    
    median_delta = df_final['delta_age'].median()
    high = df_final[df_final['delta_age'] > median_delta]
    low = df_final[df_final['delta_age'] <= median_delta]
    
    kmf.fit(high['duration'], high['event'], label=f"Predicted Older (Gap > {median_delta:.1f}y)")
    kmf.plot_survival_function(ci_show=True)
    
    kmf.fit(low['duration'], low['event'], label=f"Predicted Younger (Gap <= {median_delta:.1f}y)")
    kmf.plot_survival_function(ci_show=True)
    
    try:
        lr_res = logrank_test(high['duration'], low['duration'], event_observed_A=high['event'], event_observed_B=low['event'])
        p_val = lr_res.p_value
    except:
        p_val = 1.0
        
    plt.title(f"Survival Analysis: {model_name}\nLog-Rank p-value: {p_val:.2e}")
    plt.xlabel("Follow-up (Years)")
    plt.ylabel("Survival Probability")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"survival_{model_name}.png")
    plt.savefig(save_path)
    print(f"✅ Plot saved to {save_path}")
    
if __name__ == "__main__":
    for m in ["ecg_only", "combined", "vcg_only"]: 
        run_analysis(m)




# ## sadeer test

# import pandas as pd
# from sklearn.metrics import mean_absolute_error
# from lifelines import CoxPHFitter

# def main():
#     # 1. Load the dataset
#     # Replace with your actual csv filename
#     csv_file = "PI_mortality_data.csv" 
#     df = pd.read_csv(csv_file)

#     # Clean up column names just in case there are trailing spaces
#     df.columns = df.columns.str.strip()

#     models = ['ecg only age', 'vcg only age', 'combined age']

#     print("==================================================")
#     print("1. MEAN ABSOLUTE ERROR (MAE)")
#     print("==================================================")
#     for model in models:
#         mae = mean_absolute_error(df['true age'], df[model])
#         print(f"{model.ljust(15)}: {mae:.3f} years")

#     print("\n==================================================")
#     print("2. SURVIVAL ANALYSIS (Hazard Ratios & p-values)")
#     print("==================================================")
    
#     # Pre-process for Cox Model
#     # Convert gender to numeric (M=1, F=0) to use as a control variable
#     df['gender_num'] = df['gender'].map({'M': 1, 'F': 0})

#     # Calculate the "Age Gap" (Predicted - True). 
#     # Positive gap means their heart looks older than they actually are.
#     df['ecg_gap'] = df['ecg only age'] - df['true age']
#     df['vcg_gap'] = df['vcg only age'] - df['true age']
#     df['combined_gap'] = df['combined age'] - df['true age']

#     gap_columns = ['ecg_gap', 'vcg_gap', 'combined_gap']

#     for gap in gap_columns:
#         cph = CoxPHFitter()
        
#         # We isolate the specific gap, plus our controls (true age & gender) and survival targets
#         cols_to_keep = [gap, 'true age', 'gender_num', 'time to death or follow up', 'death']
#         model_df = df[cols_to_keep].dropna()

#         # Fit the Cox Proportional Hazards model
#         cph.fit(model_df, duration_col='time to death or follow up', event_col='death')

#         # Extract Hazard Ratio (exp(coef)) and p-value specifically for the age gap
#         hr = cph.hazard_ratios_[gap]
#         p_val = cph.summary.loc[gap, 'p']
        
#         # Format the output cleanly
#         sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#         print(f"{gap.ljust(15)}: HR = {hr:.3f} | p-value = {p_val:.3e} {sig}")

#     print("--------------------------------------------------")
#     print("Note: An HR > 1.0 indicates that a higher gap (predicted age > true age) increases mortality risk.")

# if __name__ == "__main__":
#     main()