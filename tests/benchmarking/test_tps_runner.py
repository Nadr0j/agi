import unittest

from unittest.mock import patch, Mock

from src.benchmarking.config.tps_runner_config import Config
from src.benchmarking.tps_runner import TpsRunner


# TODO: Add test constants class
class TestTpsRunnerConfig(unittest.TestCase):
    @patch.object(Config, 'config', new={
        "models": ["someModel"],
        "prompt": "somePrompt",
        "num_runs_per_model": 10
    })
    def test_load_config(self):
        runner = TpsRunner()
        runner.load_config()

        self.assertEqual({"someModel": []}, runner.results_by_model)
        self.assertEqual("somePrompt", runner.prompt)
        self.assertEqual(10, runner.num_runs_per_model)


if __name__ == '__main__':
    unittest.main()
