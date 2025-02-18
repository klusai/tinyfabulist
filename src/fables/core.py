import logging
import os
import yaml
import time
from datetime import datetime
from dotenv import load_dotenv

# The key import: Hugging Face's OpenAI-compatible client
from openai import OpenAI

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")

with open("fables/config.yml", "r") as file:
    config = yaml.safe_load(file)

SYSTEM_PROMPT = config["system_prompt"]
FABLE_PROMPT = config["fable_prompt"]

# Initialize the OpenAI-compatible client for your Hugging Face endpoint:
client = OpenAI(
    base_url=ENDPOINT_URL, 
    api_key=HF_TOKEN      
)

def generate_fable(character, trait, setting, conflict, resolution, moral):
    """
    Generates a fable using the OpenAI-compatible *Completions* endpoint.
    
    Returns a dictionary with:
      - fable_config
      - fable_prompt (the final prompt string)
      - fable_text_en (the generated text)
      - llm_name, llm_input_tokens, llm_output_tokens, etc.
    """
    # We’ll embed both system and user instructions into a single "prompt" string
    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{FABLE_PROMPT}\n"
        f"Character: {character}\n"
        f"Trait: {trait}\n"
        f"Setting: {setting}\n"
        f"Conflict: {conflict}\n"
        f"Resolution: {resolution}\n"
        f"Moral: {moral}\n"
    )

    # Meta placeholders
    llm_name = "hf_inference_model_v1"
    host_provider = "AWS"
    host_dc_provider = "AWS"
    host_dc_location = "us-east-1"
    host_gpu = "NVIDIA T4"
    host_gpu_vram = 16  # in GB
    host_cost_per_hour = 2.50
    pipeline_version = "1.0.0"

    start_time = time.time()
    logger.info("Sending request to Hugging Face OpenAI-compatible Completions endpoint.")

    # We do *completions.create*, not chat.completions
    # The model name must match what your endpoint expects, e.g. "Llama-3.1-8B-Instruct" or "tgi", etc.
    completion_stream = client.completions.create(
        model="Llama-3.1-8B-Instruct",  
        prompt=prompt_text,
        max_tokens=300,
        stream=True,        # Stream the text in chunks
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None
    )

    # Collect the streamed chunks
    generated_chunks = []
    for chunk in completion_stream:
        # For completions, the streamed text usually appears in chunk.choices[0].text
        part = chunk.choices[0].text
        generated_chunks.append(part)

    # Combine chunks into the final text
    generated_text = "".join(generated_chunks)

    end_time = time.time()
    inference_time = end_time - start_time

    # Naive token estimate for input/output
    llm_input_tokens = len(prompt_text.split())
    llm_output_tokens = len(generated_text.split())

    # Example cost formula
    llm_inference_cost_usd = (llm_input_tokens + llm_output_tokens) * 0.00001

    return {
        "fable_config": {
            "character": character,
            "trait": trait,
            "setting": setting,
            "conflict": conflict,
            "resolution": resolution,
            "moral": moral
        },
        "fable_prompt": prompt_text,
        "fable_text_en": generated_text,
        "llm_name": llm_name,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "llm_inference_time": inference_time,
        "llm_inference_cost_usd": llm_inference_cost_usd,
        "host_provider": host_provider,
        "host_dc_provider": host_dc_provider,
        "host_dc_location": host_dc_location,
        "host_gpu": host_gpu,
        "host_gpu_vram": host_gpu_vram,
        "host_cost_per_hour": host_cost_per_hour,
        "generation_datetime": datetime.utcnow().isoformat(),
        "pipeline_version": pipeline_version
    }
