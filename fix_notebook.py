import json
import re

file_path = "notebooks/regression.ipynb"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []
    in_conflict = False
    keep_lines = False

    for line in lines:
        if line.startswith("<<<<<<<"):
            in_conflict = True
            keep_lines = False  # Keep incoming/latest changes
            continue
        elif line.startswith("======="):
            keep_lines = True
            continue
        elif line.startswith(">>>>>>>"):
            in_conflict = False
            keep_lines = True
            continue

        if not in_conflict or keep_lines:
            fixed_lines.append(line)

    fixed_content = "".join(fixed_lines)
    fixed_content = re.sub(r",\s*([}\]])", r"\1", fixed_content)

    data = json.loads(fixed_content)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

    print("SUCCESS: Notebook fixed and saved successfully!")

except Exception as e:
    print(f"ERROR: {e}")
