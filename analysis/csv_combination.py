import pandas as pd
import os
import glob

def merge_csvs(folder_path, output_path=None):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if not csv_files:
        print(f"No se encontraron CSVs en {folder_path}")
        return None
    
    print(f"Encontrados {len(csv_files)} CSVs:")
    for f in csv_files:
        print(f"  {f}")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}: {len(df)} filas, {df['image_path'].nunique() if 'image_path' in df.columns else '?'} frames")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    if output_path is None:
        output_path = os.path.join(folder_path, "combined.csv")
    
    combined.to_csv(output_path, index=False)
    print(f"\nCSV combinado guardado en: {output_path}")
    print(f"Total filas: {len(combined)}")
    print(f"Total frames únicos: {combined['image_path'].nunique() if 'image_path' in combined.columns else '?'}")
    
    return combined


if __name__ == "__main__":
    FOLDER = "/mbari/Tempbox/fernanda/particles_v3/"
    OUTPUT = "/mbari/Tempbox/fernanda/particles_v3/combined.csv"  # None para guardar en la misma carpeta
    
    merge_csvs(FOLDER, OUTPUT)