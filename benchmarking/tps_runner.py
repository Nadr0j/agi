import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from requests import Response

from benchmarking.config.tps_runner_config import Config
from ollama_client.client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


# logging.getLogger().addHandler(logging.StreamHandler())


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
        raw_response: Response = client.send_request(model_name, self.prompt)
        logger.info(f"Got response from Ollama with status code {raw_response.status_code}")
        logger.info("Parsing response")

        try:
            response: Dict = json.loads(raw_response.json())
            tokens_per_second = response["eval_count"] / (response["eval_duration"] / 1000000000)
            logger.info(f"Run result for {model_name} - {tokens_per_second} t/s")
            self.results_by_model[model_name].append(tokens_per_second)
        except Exception as e:
            logger.error(f"Response parsing failed for response - {raw_response.text}")
            print(e)

    def run(self):
        self.__load_config()
        for model in list(self.results_by_model.keys()):
            self.__benchmark_model(model)

        print(self.results_by_model)

class Constants:
    MODELS: str = "models"
    PROMPT: str = "prompt"
    RUNS_PER_MODEL: str = "num_runs_per_model"
