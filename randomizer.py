import json
import random

# List of file names
file_names = [f'documentos_part_{i+1}.json' for i in range(20)]

# Randomly select a file name
file_name = random.choice(file_names)
print(f"Selected file: {file_name}")

# Load the selected JSON file
with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Randomly select 1000 entries
snippet = random.sample(data, 1000)

# Write the selected entries to a new JSON file
with open('snippet.json', 'w', encoding='utf-8') as f:
    json.dump(snippet, f, ensure_ascii=False, indent=4)