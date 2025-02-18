"""Configuration class to manage environment variables, loaded from a .env file."""

import os
from dotenv import load_dotenv

class EnvConfig:
    def __init__(self, verbose=False):
        """Initialize the EnvConfig and load environment variables."""
        load_dotenv(verbose=verbose)
        
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._hf_endpoint_url = os.getenv("HF_ENDPOINT_URL")
        self._hf_token = os.getenv("HF_TOKEN")

    @property
    def openai_api_key(self):
        return self._openai_api_key or ""

    @property
    def hf_endpoint_url(self):
        return self._hf_endpoint_url or ""

    @property
    def hf_token(self):
        return self._hf_token or ""
