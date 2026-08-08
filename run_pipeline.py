# run_pipeline.py
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_all_pipelines():
    print("🚀 Starting Production Data Atlas Processing Chain...\n")
    
    pipeline_scripts = [
        ("1/6: USAID DHS", "pipelines/ingest_usaid_dhs.py"),
        ("2/6: WHO GHO", "pipelines/ingest_who_gho.py"),
        ("3/6: MIMIC-IV", "pipelines/ingest_mimic_iv.py"),
        ("4/6: NIH All of Us", "pipelines/ingest_nih_allofus.py"),
        ("5/6: NHANES", "pipelines/ingest_nhanes.py"),
        ("6/6: UK Biobank", "pipelines/ingest_uk_biobank.py"),
    ]

    for name, script_path in pipeline_scripts:
        print(f"\n--- Running Pipeline {name} ---")
        result = subprocess.run([sys.executable, script_path])
        
        if result.returncode != 0:
            print(f"\n❌ Pipeline {name} failed with exit code: {result.returncode}")
            sys.exit(result.returncode)

    print("\n🎉 Success! All active schemas harmonized and stored in global_health_atlas.db.")

if __name__ == "__main__":
    run_all_pipelines()
    