import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../datasets/iWildCam2020")
MD_JSON = DATA_DIR / "iwildcam2020_megadetector_results.json"
METADATA_CSV = "data/metadata.csv.gz"

def get_max_animal_conf(detections):
    """Returns the highest confidence of an 'animal' (category '1') in a frame."""
    if not detections:
        return 0.0
    animal_confs = [d["conf"] for d in detections if d["category"] == "1"]
    return max(animal_confs) if animal_confs else 0.0

def main():
    print("Loading MegaDetector results...")
    with open(MD_JSON, "r") as f:
        md_data = json.load(f)
    
    # Map image_id to its max animal confidence
    conf_map = {
        img["id"]: get_max_animal_conf(img.get("detections", []))
        for img in md_data["images"]
    }

    print("Loading Metadata...")
    # Read metadata to get sequence IDs and image order
    meta = pd.read_csv(METADATA_CSV, low_memory=False)
    
    # Filter out empty annotations and map the confidences
    meta = meta[meta["category"] != "empty"].copy()
    meta["max_conf"] = meta["image_id"].map(conf_map).fillna(0.0)
    
    # Group by sequence ID
    print("Analyzing Sequences...")
    sequences = meta.groupby("seq_id")
    
    total_seqs = 0
    empty_starts = 0
    empty_ends = 0
    
    for seq_id, group in sequences:
        # Sort by filename or timestamp to ensure chronological order
        group = group.sort_values("file_name")
        
        if len(group) < 3:
            continue # Skip single-image sequences or pairs for this analysis
            
        total_seqs += 1
        confs = group["max_conf"].tolist()
        
        # Check if the sequence has an animal (max conf >= 0.6 somewhere)
        # but the boundary frames are empty (conf < 0.6)
        if max(confs) >= 0.6:
            if confs[0] < 0.6:
                empty_starts += 1
            if confs[-1] < 0.6:
                empty_ends += 1

    print("\n=== Burst Boundary Analysis ===")
    print(f"Total multi-frame sequences analyzed: {total_seqs:,}")
    print(f"Sequences starting with an empty frame: {empty_starts:,} ({empty_starts/total_seqs:.1%})")
    print(f"Sequences ending with an empty frame: {empty_ends:,} ({empty_ends/total_seqs:.1%})")

if __name__ == "__main__":
    main()