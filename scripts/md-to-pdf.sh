#!/usr/bin/bash

for model in "ollama-llama3-latest" "ollama-llava-13b" "ollama-phi:14b" "ollama::mistral:7b" "openai::gpt-4o" "openai::gpt-4o-mini" "ollama::deepseek-r1:14b" "ollama::deepseek-r1:8b"; do
    uv run prompt-refiner.py "$model" -p chat-system-1.prompt.md -e "chat-system-2" -w -i input-prompts
done