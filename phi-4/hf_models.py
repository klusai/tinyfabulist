import re
from huggingface_hub import HfApi

# Initialize the API client
api = HfApi()

# Fetch public models (limit to first 10 for debugging)
try:
    models = api.list_models(limit=100)
except Exception as e:
    print(f"Error fetching models: {e}")
    models = []

# Initialize an empty list to store metadata
model_metadata = []

# Regular expression to match parameter numbers in model names (e.g., "82M", "32B")
param_pattern = re.compile(r"(\d+[\._]?\d*[MBmb])")

# Loop through models to extract metadata
for model in models:
    try:
        model_info = api.model_info(model.modelId)

        # Extract parameters from model name if available
        match = param_pattern.search(model.modelId)
        parameters = match.group(0) if match and (match.group(0).endswith("B") or match.group(0).endswith("M") or match.group(0).endswith("m") or match.group(0).endswith("b")) else "N/A"

        # Collect metadata with safe handling for None values
        metadata = {
            "model_id": str(model.modelId) if model.modelId else "N/A",
            "downloads": str(model_info.downloads) if model_info.downloads else "0",
            "parameters": parameters,
            "last_updated": str(model_info.lastModified) if model_info.lastModified else "N/A",
            "license": model_info.cardData.get("license", "N/A") if model_info.cardData else "N/A",
        }
        model_metadata.append(metadata)
    except Exception as e:
        print(f"Error fetching metadata for {model.modelId}: {e}")

# Print the collected metadata
print(f"{'Model ID':<40}{'Downloads':<15}{'Parameters':<15}{'Last Updated':<25}{'License':<15}")
print("=" * 120)
for meta in model_metadata:
    # Ensure every field is non-None and converted to a string
    model_id = str(meta.get('model_id', "N/A") or "N/A")
    downloads = str(meta.get('downloads', "0") or "0")
    parameters = str(meta.get('parameters', "N/A") or "N/A")
    last_updated = str(meta.get('last_updated', "N/A") or "N/A")
    license_info = str(meta.get('license', "N/A") or "N/A")

    print(
        f"{model_id:<40}"
        f"{downloads:<15}"
        f"{parameters:<15}"
        f"{last_updated:<25}"
        f"{license_info:<15}"
    )
