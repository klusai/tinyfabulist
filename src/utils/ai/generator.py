import time
import logging
from datetime import datetime
from openai import OpenAI  # Hugging Face's OpenAI-compatible client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerativeAICore:
    """
    A class to generate fables using a Hugging Face OpenAI-compatible completions endpoint.
    """

    def __init__(self, system_prompt, fable_prompt, endpoint_url, api_key, model="Llama-3.1-8B-Instruct"):
        """
        Initializes the AIFableGenerator with required prompts, endpoint, and credentials.
        
        Args:
            system_prompt (str): The system prompt to prepend to all requests.
            fable_prompt (str): The fable-specific prompt.
            endpoint_url (str): The URL of the Hugging Face inference endpoint.
            api_key (str): The API key for authentication.
            model (str): The model name to use (default: "Llama-3.1-8B-Instruct").
        """
        self.__SYSTEM_PROMPT = system_prompt
        self.__FABLE_PROMPT = fable_prompt
        self.model = model
        self.client = OpenAI(base_url=endpoint_url, api_key=api_key)

    def generate_fable(self, character, trait, setting, conflict, resolution, moral):
        """
        Generates a fable based on provided parameters.

        Args:
            character (str): The main character of the fable.
            trait (str): The trait of the character.
            setting (str): The setting of the fable.
            conflict (str): The conflict within the story.
            resolution (str): How the conflict is resolved.
            moral (str): The moral of the fable.

        Returns:
            dict: A dictionary containing the generated fable and its metadata.
        """
        prompt_text = (
            f"{self.__SYSTEM_PROMPT}\n\n"
            f"{self.__FABLE_PROMPT}\n"
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
        logger.info("Sending request to Hugging Face OpenAI-compatible completions endpoint.")

        # The model name must match what your endpoint expects.
        completion_stream = self.client.completions.create(
            model=self.model,
            prompt=prompt_text,
            max_tokens=300,
            stream=True,  # Stream the text in chunks
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
                "moral": moral,
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


