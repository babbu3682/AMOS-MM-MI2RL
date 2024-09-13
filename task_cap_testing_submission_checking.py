import json

def load_json(file_path):
    """Load and return the JSON data from the provided file path."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def check_test_structure(data):
    """Check the structure of the 'testing' key in the JSON data."""
    if 'testing' not in data:
        print("Missing 'testing' key in JSON data.")
        return False

    test_data = data['testing']
    if not isinstance(test_data, list):
        print("The 'validation' key should contain a list of dictionaries.")
        return False

    # Check each item in the validation data list
    for item in test_data:
        if not isinstance(item, dict):
            print("Each item in the 'test' list should be a dictionary.")
            return False
        if 'labels' not in item:
            print("Each testing item should have a 'labels' key.")
            return False
        if 'report' not in item['labels']:
            print("Each 'labels' dictionary should contain a 'report' key.")
            return False
        report = item['labels']['report']
        if 'findings' not in report:
            print("Each 'report' should have a 'findings' key.")
            return False
        findings = report['findings']
        for key in ['chest', 'abdomen', 'pelvis']:
            if key not in findings:
                print(f"Missing '{key}' key in 'findings'.")
                return False
            if not isinstance(findings[key], str):
                print(f"The '{key}' value in 'findings' should be a string.")
                return False
    return True

def display_expected_format():
    """Display the expected format of the JSON file."""
    print("""
Expected JSON Format:
{
    "testing": [
        {
            "image": "path/to/image/file",
            "labels": {
                "report": {
                    "findings": {
                        "chest": "string",
                        "abdomen": "string",
                        "pelvis": "string"
                    },
                }
            }
        },
        ... (additional testing cases)
    ]
}
""")

def main(file_path):
    data = load_json(file_path)
    if data is None:
        return
    
    # Display expected format
    display_expected_format()

    # Perform the checks
    if check_test_structure(data):
        print("The JSON file meets all the requirements.")
    else:
        print("The JSON file does not meet the requirements.")
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_submission.py <path_to_json_file>")
    else:
        main(sys.argv[1])
