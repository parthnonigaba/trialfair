"""
Convert AACT and EU CSV files to SQLite for instant lookups.

Run in your trialfair_deploy/final_data folder:
    python convert_to_sqlite.py
"""

import sqlite3
import pandas as pd
import os

DB_FILE = "trialfair.db"

def convert_csv_to_sqlite():
    conn = sqlite3.connect(DB_FILE)
    
    # AACT trials
    if os.path.exists("aact_master.csv"):
        print("Loading aact_master.csv...")
        df = pd.read_csv("aact_master.csv", low_memory=False)
        # Keep only rows with eligibility text
        df = df[df['eligibility_text'].notna() & (df['eligibility_text'].str.len() > 50)]
        print(f"  {len(df):,} AACT trials with eligibility text")
        
        df.to_sql("aact_trials", conn, if_exists="replace", index=False)
        
        # Create index on nct_id for fast lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_aact_nct_id ON aact_trials(nct_id)")
        print("  ✓ Created aact_trials table with index")
    else:
        print("⚠ aact_master.csv not found")
    
    # EU trials
    eu_file = None
    for f in ["eu_master.csv", "eu_with_rindex_and_text.csv"]:
        if os.path.exists(f):
            eu_file = f
            break
    
    if eu_file:
        print(f"Loading {eu_file}...")
        df = pd.read_csv(eu_file, low_memory=False)
        # Keep only rows with eligibility text
        if 'eligibility_text' in df.columns:
            df = df[df['eligibility_text'].notna() & (df['eligibility_text'].str.len() > 50)]
        print(f"  {len(df):,} EU trials with eligibility text")
        
        df.to_sql("eu_trials", conn, if_exists="replace", index=False)
        
        # Create index on eudract_number for fast lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eu_eudract ON eu_trials(eudract_number)")
        print("  ✓ Created eu_trials table with index")
    else:
        print("⚠ EU CSV file not found")
    
    conn.commit()
    conn.close()
    
    # Show file size
    size_mb = os.path.getsize(DB_FILE) / 1024 / 1024
    print(f"\n✓ Created {DB_FILE} ({size_mb:.1f} MB)")
    print("\nYou can now delete the large CSV files if you want to save space.")

if __name__ == "__main__":
    convert_csv_to_sqlite()
