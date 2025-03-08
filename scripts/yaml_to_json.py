#!/usr/bin/env python3
import yaml
import json
import sys
import os

def yaml_to_json(yaml_file, json_file):
    """
    Convert YAML file to JSON file
    
    Args:
        yaml_file (str): Path to the input YAML file
        json_file (str): Path to the output JSON file
    """
    try:
        # Check if input file exists
        if not os.path.isfile(yaml_file):
            print(f"Error: Input file '{yaml_file}' does not exist.")
            return False
        
        # Read YAML file
        with open(yaml_file, 'r') as yf:
            # Parse YAML into Python dictionary
            yaml_content = yaml.safe_load(yf)
        
        # Write JSON file
        with open(json_file, 'w') as jf:
            # Convert Python dictionary to JSON and write to file
            json.dump(yaml_content, jf, indent=2)
        
        print(f"Successfully converted '{yaml_file}' to '{json_file}'")
        return True
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python yaml_to_json.py <yaml_file> <json_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    json_file = sys.argv[2]
    
    # Convert YAML to JSON
    success = yaml_to_json(yaml_file, json_file)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()