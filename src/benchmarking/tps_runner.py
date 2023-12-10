import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import requests
from requests import Response

from src.benchmarking.config.tps_runner_config import Config
from src.ollama_client.client import Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TpsRunner:
    results_by_model: Dict = field(default_factory=dict)
    prompt: Optional[str] = None
    num_runs_per_model: Optional[int] = None

    def __load_config(self):
        logger.info("Loading TpsRunner config")
        self.results_by_model = {model: list() for model in Config.config[Constants.MODELS]}
        logger.info(f"Running benchmarking for models - {Config.config[Constants.MODELS]}")
        self.prompt = Config.config[Constants.PROMPT]
        logger.info(f"Running benchmarking using prompt - {self.prompt}")
        self.num_runs_per_model = Config.config[Constants.RUNS_PER_MODEL]
        logger.info(f"Running benchmarking number of times - {self.num_runs_per_model}")

    def __benchmark_model(self, model_name: str):
        # TODO: Replace this with check if model is already pulled
        logger.info(f"Pulling {model_name}")
        os.system(f"ollama pull {model_name}")

        client: Client = Client()
        logger.info(f"Making request to client for model {model_name}")
        response: Response = client.send_request(model_name, self.prompt)
        print(response.text)

    def run(self):
        self.__load_config()
        for model in list(self.results_by_model.keys()):
            self.__benchmark_model(model)


class Constants:
    MODELS: str = "models"
    PROMPT: str = "prompt"
    RUNS_PER_MODEL: str = "num_runs_per_model"
