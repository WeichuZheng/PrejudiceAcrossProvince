import json
import os
from typing import List, Dict, Any, Optional

def save_dict_to_jsonl(data_dict: Dict[str, Any], filepath: str, mode: str = 'a') -> None:
    """
    Save a single dictionary as one line in a JSONL file.
    
    Args:
        data_dict: Dictionary to save
        filepath: Path to the JSONL file
        mode: File mode ('a' for append, 'w' for write/overwrite)
    """
    with open(filepath, mode, encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)
        f.write('\n')

def save_dicts_to_jsonl(data_list: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save multiple dictionaries to a JSONL file.
    
    Args:
        data_list: List of dictionaries to save
        filepath: Path to the JSONL file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for data_dict in data_list:
            json.dump(data_dict, f, ensure_ascii=False)
            f.write('\n')

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load all records from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries
    """
    records = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def update_record_by_image_path(filepath: str, image_path: str, updates: Dict[str, Any]) -> bool:
    """
    Update a specific record in JSONL file by matching 'image_path' attribute.
    
    Args:
        filepath: Path to the JSONL file
        image_path: The image_path value to match
        updates: Dictionary with updates to apply
        
    Returns:
        True if record was found and updated, False otherwise
    """
    records = load_jsonl(filepath)
    record_found = False
    
    # Find and update the record
    for record in records:
        if record.get('image_path') == image_path:
            record.update(updates)
            record_found = True
            break
    
    if record_found:
        # Rewrite the entire file with updated records
        save_dicts_to_jsonl(records, filepath)
        return True
    return False

def find_record_by_image_path(filepath: str, image_path: str) -> Optional[Dict[str, Any]]:
    """
    Find a specific record by image_path.
    
    Args:
        filepath: Path to the JSONL file
        image_path: The image_path value to match
        
    Returns:
        The matching record dictionary or None if not found
    """
    records = load_jsonl(filepath)
    for record in records:
        if record.get('image_path') == image_path:
            return record
    return None