import json

# Load the original JSON file
with open('documentos.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Calculate the size of each part
part_size = len(data) // 20

# Split the data into 20 parts
for i in range(20):
    start = i * part_size
    end = (i + 1) * part_size if i < 19 else len(data)
    part_data = data[start:end]

    # Write each part to a new JSON file
    with open(f'documentos_part_{i+1}.json', 'w', encoding='utf-8') as f:
        json.dump(part_data, f, ensure_ascii=False, indent=4)