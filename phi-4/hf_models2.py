from huggingface_hub import HfApi

api = HfApi()
model_id = "distilbert-base-uncased"  # Known to have Apache-2.0 license
info = api.model_info(model_id)

print("Top-level license:", info.license)
print("cardData license:", info.cardData.get("license") if info.cardData else None)
