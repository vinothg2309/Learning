
# NeMo Guardrails: Installation and Basics



## Overview

NeMo Guardrails is an open-source toolkit for adding programmable guardrails to LLM-based conversational systems. It provides:
**Input Rails**: Control what users can ask
**Output Rails**: Control what the LLM responds with
**Retrieval Rails**: Control what information is retrieved from knowledge bases
**Dialog Rails**: Control the flow of conversation


## Key Concepts

1. **Colang**: A modeling language for conversational flows
2. **Rails**: Safety and control mechanisms
3. **Actions**: Custom Python functions that can be executed
4. **Configuration**: YAML-based setup for guardrails


## Installation



# Install NeMo Guardrails

!pip install nemoguardrails


# Install additional dependencies for OpenAI-compatible endpoints

!pip install openai langchain langchain-openai


## Import Required Libraries


import os
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model
import yaml
import nest_asyncio


# Apply nest_asyncio to allow sync generate() calls in Jupyter

nest_asyncio.apply()


## Configure Custom LLM Endpoint


NeMo Guardrails supports OpenAI-compatible endpoints. We'll configure it to use your custom endpoint.


# API Configuration

[REDACTED_API_KEY]
BASE_URL = ""


# Set environment variables for OpenAI-compatible endpoint

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


## Basic Configuration Structure


NeMo Guardrails uses a YAML configuration file with the following structure:
`config.yml`: Main configuration file
`config.co`: Colang file for conversation flows
`actions.py`: Custom Python actions (optional)


## Example 1: Simple Guardrails Configuration


Let's create a basic configuration with no rails to understand the structure.


# Basic configuration dictionary

basic_config = {
"models": [
{
"type": "main",
"engine": "openai",
"model": "gpt-4o-mini",
"parameters": {
"temperature": 0.0,
"max_tokens": 256
}
}
],
"instructions": [
{
"type": "general",
"content": "You are a helpful AI assistant. Answer questions accurately and concisely."
}
]
}


# Convert to YAML

print(yaml.dump(basic_config, default_flow_style=False))

Output:
instructions:
content: You are a helpful AI assistant. Answer questions accurately and concisely.
type: general
models:
engine: openai
model: gpt-4o-mini
parameters:
max_tokens: 256
temperature: 0.0
type: main


## Example 2: Initialize RailsConfig


There are two ways to initialize RailsConfig:
1. From a directory containing config files
2. From a dictionary (programmatic approach)


# Method 1: Programmatic configuration

config = RailsConfig.from_content(
yaml_content=yaml.dump(basic_config),
colang_content="""

# Basic greeting flow

define user express greeting
"hello"
"hi"
"hey"

define bot express greeting
"Hello! How can I help you today?"

define flow greeting
user express greeting
bot express greeting
"""
)


# Initialize LLMRails

rails = LLMRails(config)

print("✓ Guardrails initialized successfully!")

Output:
/Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
from .autonotebook import tqdm as notebook_tqdm

✓ Guardrails initialized successfully!


## Example 3: Test Basic Interaction



# Test the guardrails

response = rails.generate(messages=[{
"role": "user",
"content": "Hello!"
}])

print(f"User: Hello!")
print(f"Bot: {response['content']}")

Output:
User: Hello!
Bot: Hello! How can I help you today?


# Test with a regular question

response = rails.generate(messages=[{
"role": "user",
"content": "What is machine learning?"
}])

print(f"User: What is machine learning?")
print(f"Bot: {response['content']}")

Output:
User: What is machine learning?
Bot: Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data. Instead of being explicitly programmed for specific tasks, machine learning systems improve their performance as they are exposed to more data over time.


## Example 4: Configuration File Structure


Let's save a proper configuration to files.


# Create a basic config directory

!mkdir -p configs/basic_bot


# Write config.yml

config_yml = """
models:
- type: main
engine: openai
model: gpt-4o-mini
parameters:
temperature: 0.0
max_tokens: 256

instructions:
- type: general
content: |
You are a helpful AI assistant.
Answer questions accurately and concisely.
Be polite and professional.
"""

with open('configs/basic_bot/config.yml', 'w') as f:
f.write(config_yml)

print("✓ config.yml created")

Output:
✓ config.yml created


# Write config.co (Colang file)

config_co = """

# Define user intents

define user express greeting
"hello"
"hi"
"hey there"
"good morning"
"good afternoon"

define user ask about capabilities
"what can you do"
"what are your capabilities"
"how can you help me"


# Define bot responses

define bot express greeting
"Hello! How can I help you today?"

define bot inform capabilities
"I can answer questions, provide information, and assist with various tasks. What would you like to know?"


# Define flows

define flow greeting
user express greeting
bot express greeting

define flow capabilities
user ask about capabilities
bot inform capabilities
"""

with open('configs/basic_bot/config.co', 'w') as f:
f.write(config_co)

print("✓ config.co created")

Output:
✓ config.co created


## Example 5: Load Configuration from Directory



# Load configuration from directory

config_from_dir = RailsConfig.from_path("configs/basic_bot")
rails_from_dir = LLMRails(config_from_dir)

print("✓ Configuration loaded from directory")

Output:
✓ Configuration loaded from directory


# Test the bot

test_messages = [
"Hello!",
"What can you do?",
"What is the capital of France?"
]

for message in test_messages:
response = rails_from_dir.generate(messages=[{
"role": "user",
"content": message
}])
print(f"\nUser: {message}")
print(f"Bot: {response['content']}")

Output:

User: Hello!
Bot: Hello! How can I help you today?

User: What can you do?
Bot: I can answer questions, provide information, and assist with various tasks. What would you like to know?

User: What is the capital of France?
Bot: The capital of France is Paris.


## Understanding the Response Object


The `generate()` method returns a dictionary with useful information.


# Get detailed response

response = rails_from_dir.generate(
messages=[{"role": "user", "content": "Hello!"}]
)

print("Response keys:", response.keys())
print("\nContent:", response['content'])


# The response object contains the generated content and metadata


# You can inspect all available keys in the response

print("\nAvailable response fields:")
for key in response.keys():
print(f"  - {key}")

Output:
Response keys: dict_keys(['role', 'content'])

Content: Hello! How can I help you today?

Available response fields:
- role
- content


## Multi-turn Conversations



# Multi-turn conversation

conversation = [
{"role": "user", "content": "Hello!"},
]

response = rails_from_dir.generate(messages=conversation)
print(f"User: {conversation[0]['content']}")
print(f"Bot: {response['content']}\n")


# Add bot response to conversation

conversation.append({"role": "assistant", "content": response['content']})


# Continue conversation

conversation.append({"role": "user", "content": "What's the weather like?"})
response = rails_from_dir.generate(messages=conversation)
print(f"User: {conversation[-1]['content']}")
print(f"Bot: {response['content']}")

Output:
User: Hello!
Bot: Hello! How can I help you today?

User: What's the weather like?
Bot: I don't have real-time weather data, but you can check a weather website or app for the latest updates. If you need help with anything else, feel free to ask!


## Summary


In this notebook, we covered:
1. **Installation**: Installing NeMo Guardrails and dependencies
2. **Configuration**: Setting up custom OpenAI-compatible endpoints
3. **Basic Structure**: Understanding config.yml and config.co files
4. **Initialization**: Two methods to create RailsConfig
5. **Testing**: Running basic conversations


## Next Steps

**Notebook 02**: Input Rails (Jailbreak prevention, topic control, moderation)
**Notebook 03**: Output Rails (Fact checking, hallucination detection)
**Notebook 04**: Retrieval Rails (RAG integration)
**Notebook 05**: Dialog Rails and Custom Actions
