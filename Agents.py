import openai
import requests
import json
from typing import List, Dict, Any


# In ra phản hồi


class OllamaPromptEngineer:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def basic_prompt(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Basic prompt execution with Ollama"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        response = requests.post(url, json=payload)
        return response.json()["response"]

    def system_prompt(self, model: str, system: str, user_prompt: str) -> str:
        """Execute prompt with system context"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        response = requests.post(url, json=payload)
        return response.json()["message"]["content"]

    def few_shot_prompt(self, model: str, task_description: str,
                        examples: List[Dict[str, str]], query: str) -> str:
        """Implement few-shot prompting"""
        prompt = f"{task_description}\n\nExamples:\n"

        for example in examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"

        prompt += f"Input: {query}\nOutput:"

        return self.basic_prompt(model, prompt)

    def code_generation_prompt(self, model: str, language: str,
                               task: str, requirements: List[str] = None) -> str:
        """Specialized code generation prompting"""
        system_prompt = f"""You are an expert {language} programmer. 
        Write clean, efficient, and well-documented code. 
        Follow best practices and include error handling where appropriate."""

        user_prompt = f"Task: {task}\n"
        if requirements:
            user_prompt += f"Requirements:\n" + "\n".join(f"- {req}" for req in requirements)

        return self.system_prompt(model, system_prompt, user_prompt)

    def structured_output_prompt(self, model: str, data_request: str,
                                 output_format: str = "json") -> str:
        """Request structured output format"""
        system_prompt = f"""Always respond with valid {output_format.upper()} format. 
        Do not include any explanatory text outside the {output_format} structure."""

        return self.system_prompt(model, system_prompt, data_request)


if __name__ == "__main__":
    engineer = OllamaPromptEngineer()

    # Basic prompting
    basic_response = engineer.basic_prompt(
        model="dolphin3",
        prompt="Explain quantum computing in simple terms"
    )
    print("Basic Prompt Response:", basic_response)

    # System prompt example
    sql_response = engineer.system_prompt(
        model="dolphin3",
        system="You are a SQL expert. Always provide optimized queries with explanations.",
        user_prompt="Create a query to find the top 5 customers by purchase amount"
    )
    print("System Prompt Response:", sql_response)

    # Few-shot prompting for sentiment analysis
    sentiment_examples = [
        {"input": "I love this product!", "output": "positive"},
        {"input": "This is terrible quality", "output": "negative"},
        {"input": "It's okay, nothing special", "output": "neutral"}
    ]

    sentiment_result = engineer.few_shot_prompt(
        model="dolphin3",
        task_description="Classify the sentiment of the following text as positive, negative, or neutral:",
        examples=sentiment_examples,
        query="The service was amazing and exceeded my expectations"
    )
    print("Few-shot Response:", sentiment_result)

    # Code generation with requirements
    code_response = engineer.code_generation_prompt(
        model="dolphin3",
        language="Python",
        task="Create a function to calculate fibonacci numbers",
        requirements=[
            "Use memoization for efficiency",
            "Handle edge cases (n <= 0)",
            "Include docstring with examples"
        ]
    )
    print("Code Generation Response:", code_response)

    # Structured output example
    structured_response = engineer.structured_output_prompt(
        model="dolphin3",
        data_request="Extract key information about Python programming language including creator, year created, and main use cases",
        output_format="json"
    )
    print("Structured Output Response:", structured_response)