import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
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
            self.results_by_model[model_name].append(raw_response.json())
        except Exception as e:
            logger.error(f"Response parsing failed for response - {raw_response.text}")
            logger.error(e)

    def __get_average_tokens_per_second(self, model_name: str):
        benchmark_results = []
        for benchmark_run in self.results_by_model[model_name]:
            benchmark_results.append(benchmark_run["eval_count"] / (benchmark_run["eval_duration"] / 1000000000))
        return sum(benchmark_results) / len(benchmark_results)

    def run(self):
        # Callable must take model name as arg
        # TODO: Instead of callable use an abstract class or something less hacky
        analyses: list[Callable] = [self.__get_average_tokens_per_second]

        self.__load_config()
        for model in list(self.results_by_model.keys()):
            for _ in range(self.num_runs_per_model):
                logger.info(f"Making call number [{_ + 1}] for {model}")
                self.__benchmark_model(model)
            logger.info(f"Model {model} average tokens-per-second {self.__get_average_tokens_per_second(model)}")

        logger.info("Benchmarking complete. Calculating statistics")
        for model in list(self.results_by_model.keys()):
            logging.info(f"########## STATISTICS FOR [{model}] ##########")
            for callable_ in analyses:
                callable_(model)


class Constants:
    MODELS: str = "models"
    PROMPT: str = "prompt"
    RUNS_PER_MODEL: str = "num_runs_per_model"
