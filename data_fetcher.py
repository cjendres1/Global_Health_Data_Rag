import pandas as pd
import requests

def fetch_live_data(variable_id: str, table_id: str, source_dataset: str, limit: int = 50) -> pd.DataFrame:
    """Routes an individual variable ID to its respective health data endpoint."""
    source = str(source_dataset).lower().strip()
    var_id = str(variable_id).strip()
    
    try:
        # 1. WHO Global Health Observatory (Public OData API)
        if "who" in source or "gho" in source:
            url = f"https://ghoapi.azureedge.net/api/{var_id}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("value", [])
                if data:
                    df = pd.DataFrame(data)
                    df["variable_id"] = var_id
                    df["source_dataset"] = source_dataset
                    cols = [c for c in ["variable_id", "source_dataset", "SpatialDim", "TimeDim", "NumericValue", "Value"] if c in df.columns]
                    return df[cols].head(limit)

        # 2. USAID DHS Program (Public REST API)
        elif "dhs" in source or "usaid" in source:
            url = f"https://api.dhsprogram.com/rest/dhs/data?indicatorIds={var_id}&f=json"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("Data", [])
                if data:
                    df = pd.DataFrame(data)
                    df["variable_id"] = var_id
                    df["source_dataset"] = source_dataset
                    cols = [c for c in ["variable_id", "source_dataset", "CountryName", "SurveyYear", "Value", "CharacteristicLabel"] if c in df.columns]
                    return df[cols].head(limit)

        # 3. CDC NHANES (CDC Web Direct / XPT Transport Files)
        elif "nhanes" in source:
            table_name = table_id.upper() if table_id and table_id != "N/A" else "DEMO_J"
            xpt_url = f"https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{table_name}.XPT"
            df = pd.read_sas(xpt_url, format="xport")
            if var_id in df.columns:
                sub_df = df[["SEQN", var_id]].dropna().head(limit).copy()
                sub_df.rename(columns={var_id: "value"}, inplace=True)
                sub_df["variable_id"] = var_id
                sub_df["source_dataset"] = "NHANES"
                return sub_df[["variable_id", "source_dataset", "SEQN", "value"]]

    except Exception:
        pass

    return pd.DataFrame()


def fetch_all_live_data(records: list, limit_per_var: int = 25) -> pd.DataFrame:
    """Iterates through top RAG match records and fetches observation data into a unified DataFrame."""
    all_frames = []
    
    for rec in records:
        v_id = rec.get("variable_id", "")
        t_id = rec.get("table_id", "")
        s_ds = rec.get("source_dataset", "")
        
        if v_id and v_id != "N/A":
            df_single = fetch_live_data(v_id, t_id, s_ds, limit=limit_per_var)
            if not df_single.empty:
                all_frames.append(df_single)
                
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return pd.DataFrame()