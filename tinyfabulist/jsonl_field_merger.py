#!/usr/bin/env python3
import argparse
import json
import os
from tqdm import tqdm

def merge_jsonl_field(source_file, target_file, field_name, output_file=None):
    """
    Merge a field from one JSONL file into another JSONL file.
    
    Args:
        source_file (str): Path to the source JSONL file containing the field to extract
        target_file (str): Path to the target JSONL file to add the field to
        field_name (str): Name of the field to copy from source to target
        output_file (str, optional): Path to save the result. If None, will modify target_file
    
    Returns:
        int: Number of processed lines
    """
    # If no output file is specified, we'll create a temporary file and then replace the target
    if not output_file:
        output_file = f"{target_file}.tmp"
        replace_target = True
    else:
        replace_target = False
    
    # Read all source fields first
    source_fields = []
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if field_name in data:
                        source_fields.append(data[field_name])
                    else:
                        # If field is missing, add None as placeholder
                        source_fields.append(None)
                except json.JSONDecodeError:
                    # Handle invalid JSON
                    source_fields.append(None)
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found")
        return 0
    
    # Get the count of source entries
    source_count = len(source_fields)
    if source_count == 0:
        print(f"Error: No entries found in source file '{source_file}'")
        return 0
    
    # Process the target file line by line and add the field
    processed_count = 0
    try:
        with open(target_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Process with a progress bar
            with tqdm(total=source_count, desc="Merging files") as pbar:
                for i, line in enumerate(infile):
                    try:
                        # Stop if we've reached the end of source fields
                        if i >= source_count:
                            break
                        
                        # Parse the JSON line from target file
                        data = json.loads(line.strip())
                        
                        # Add the field from source file
                        source_value = source_fields[i]
                        if source_value is not None:
                            data[field_name] = source_value
                        
                        # Write the updated JSON to output file
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        processed_count += 1
                        pbar.update(1)
                        
                    except json.JSONDecodeError:
                        # Write the original line if we can't parse it
                        outfile.write(line)
    except FileNotFoundError:
        print(f"Error: Target file '{target_file}' not found")
        return 0
    
    # Replace target file with the updated version if needed
    if replace_target:
        os.replace(output_file, target_file)
        print(f"Updated {target_file} with merged field '{field_name}'")
    else:
        print(f"Saved merged result to {output_file}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Merge a field from one JSONL file into another")
    parser.add_argument("source_file", help="Source JSONL file containing the field to extract")
    parser.add_argument("target_file", help="Target JSONL file to add the field to")
    parser.add_argument("field_name", default="fable", help="Name of the field to copy from source to target")
    parser.add_argument("--output", "-o", help="Path to save the output (default: overwrite target file)")
    
    args = parser.parse_args()
    
    processed = merge_jsonl_field(
        args.source_file, 
        args.target_file, 
        args.field_name,
        args.output
    )
    
    print(f"Successfully processed {processed} entries")

if __name__ == "__main__":
    main() 