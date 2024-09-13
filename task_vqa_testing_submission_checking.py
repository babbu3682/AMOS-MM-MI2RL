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

def check_qa_structure(data):
    """Check the structure of the 'qa' key in the JSON data."""
    if 'testing' not in data:
        print("Missing 'testing' key in JSON data.")
        return False

    testing_data = data['testing']
    if not isinstance(testing_data, list):
        print("The 'testing' key should contain a list of dictionaries.")
        return False

    for item in testing_data:
        if not isinstance(item, dict):
            print("Each item in the 'testing' list should be a dictionary.")
            return False
        if 'labels' not in item:
            print("Each testing item should have a 'labels' key.")
            return False
        labels = item['labels']
        if 'qa' not in labels:
            print("Each 'labels' dictionary should contain a 'qa' key.")
            return False
        qa_list = labels['qa']
        if not isinstance(qa_list, list):
            print("The 'qa' key should contain a list of dictionaries.")
            return False
        for qa in qa_list:
            required_keys = ['question', 'prediction']
            if not all(key in qa for key in required_keys):
                print(f"Missing keys in QA item, expected keys are: {required_keys}")
                return False
            if qa['prediction'] not in ['A', 'B', 'C', 'D']:
                print("The 'prediction' must be one of the uppercase letters: A, B, C, or D.")
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
                            "qa": [
                                {
                                    "question": "String",
                                    "options": {
                                        "A": "String",
                                        "B": "String",
                                        ...
                                    },
                                    "prediction": "A",
                                    "type": "String"
                                },
                                ... (additional QA cases)
                            ]
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
    if check_qa_structure(data):
        print("The JSON file meets all the requirements.")
    else:
        print("The JSON file does not meet the requirements.")
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_submission.py <path_to_json_file>")
    else:
        main(sys.argv[1])
