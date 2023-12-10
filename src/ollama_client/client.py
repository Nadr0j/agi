from dataclasses import dataclass, field
from typing import Dict

import requests


@dataclass
class Client:
    url: str = "http://localhost:11434/api/generate"
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})

    def send_request(self, model, prompt) -> requests.Response:
        data: Dict = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.url, json=data, headers=self.headers)
        return response
