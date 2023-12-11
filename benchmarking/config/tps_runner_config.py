class Config:
    config = {
        "models": ["mistral:7b-text-q4_K_M", "codellama:7b-instruct-q4_K_M", "llama2:70b-chat-q4_K_M"],
        "prompt": "Please list 50 unique english words.",
        "num_runs_per_model": 5
    }
