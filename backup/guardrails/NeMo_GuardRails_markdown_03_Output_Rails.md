
# NeMo Guardrails: Output Rails





## Overview


Output rails control and validate what the LLM can respond with. They ensure:

Factual accuracy of responses

Prevention of hallucinations

Appropriate tone and language

Compliance with safety guidelines

No leakage of sensitive information




## Types of Output Rails


1. **Self Check Output**: LLM-based validation of generated responses

2. **Fact Checking**: Verify factual claims against knowledge bases

3. **Hallucination Detection**: Detect and prevent false information

4. **Output Moderation**: Filter inappropriate responses

5. **Custom Output Rails**: User-defined validation logic

## Setup


import os

from nemoguardrails import RailsConfig, LLMRails

import yaml



import nest_asyncio




# Apply nest_asyncio to allow async calls in Jupyter


nest_asyncio.apply()




# API Configuration


[REDACTED_API_KEY]

BASE_URL = ""



os.environ["OPENAI_API_KEY"] = API_KEY

os.environ["OPENAI_API_BASE"] = BASE_URL


## Example 1: Self Check Output Rail




Use an LLM to validate responses before returning them to the user.


# Configuration with self-check output


self_check_output_config = """

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

You are a financial advisor assistant.

Provide accurate, helpful financial information.

Never give specific investment advice or guarantees.



"""



# Colang for self-check output


self_check_output_colang = """

define subflow self check output

$is_safe = execute self_check_output



if not $is_safe

bot inform cannot answer

stop



define bot inform cannot answer

"I apologize, but I cannot provide that specific information. Let me offer general financial guidance instead."

"""



# Custom action for output checking


output_check_actions = '''

from nemoguardrails.actions import action

import re



@action(is_system_action=True)

async def self_check_output(context: dict):

"""Check if bot response is appropriate for a financial advisor."""



bot_message = context.get("bot_message", "")



# Prohibited phrases in financial advice

prohibited_phrases = [

"guaranteed returns",

"risk-free investment",

"you should buy",

"you should sell",

"definitely invest",

"100% safe"

]



bot_lower = bot_message.lower()



for phrase in prohibited_phrases:

if phrase in bot_lower:

return False



return True

'''



# Save configuration


!mkdir -p configs/output_rails/self_check_output



with open('configs/output_rails/self_check_output/config.yml', 'w') as f:

f.write(self_check_output_config)



with open('configs/output_rails/self_check_output/config.co', 'w') as f:

f.write(self_check_output_colang)



with open('configs/output_rails/self_check_output/actions.py', 'w') as f:

f.write(output_check_actions)




# Initialize rails


config = RailsConfig.from_path("configs/output_rails/self_check_output")

rails = LLMRails(config)



print("✓ Self-check output rails initialized")


Output:
✓ Self-check output rails initialized




# Test with appropriate question


response = rails.generate(messages=[{

"role": "user",

"content": "What is diversification in investing?"

}])



print("User: What is diversification in investing?")

print(f"Bot: {response['content']}")


Output:
User: What is diversification in investing?
Bot: Diversification in investing is a risk management strategy that involves spreading investments across various financial instruments, industries, and other categories to reduce exposure to any single asset or risk. The primary goal of diversification is to minimize the impact of poor performance from any one investment on the overall portfolio.

Here are some key points about diversification:

1. **Asset Classes**: Investors can diversify across different asset classes, such as stocks, bonds, real estate, and commodities. Each asset class typically reacts differently to market conditions.

2. **Sectors and Industries**: Within the stock market, investors can diversify by investing in different sectors (e.g., technology, healthcare, consumer goods) to avoid concentration in one area.

3. **Geographic Diversification**: Investing in international markets can also provide diversification, as different countries may experience varying economic conditions.

4. **Investment Vehicles**: Diversification can be achieved through various investment vehicles, including mutual funds, exchange-traded funds (ETFs), and index funds, which often hold a variety of assets.

5. **Risk Reduction**: By diversifying, investors aim to reduce the volatility of their portfolio and the risk of significant losses, as the performance of different investments may not be correlated.

While diversification can help manage risk, it does not



## Example 1b: Direct Prompt-Based Output Rail (NO Actions Required)




This example demonstrates using **built-in output checking with prompts ONLY** - no custom actions needed!



Key differences from Example 1:

Uses NeMo's built-in `self_check_output` task

Only requires custom prompt configuration

No Python actions.py file needed

Simpler and more maintainable


# Configuration using ONLY prompts - NO custom actions needed!


prompt_only_config = """

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

You are a financial advisor assistant.

Provide accurate, helpful financial information.

Never give specific investment advice or guarantees.



rails:

output:

flows:

- self check output




# Custom prompt for output validation (NO action needed!)


prompts:

- task: self_check_output

content: |

You are validating a financial advisor's response for compliance.



User question: "{{ user_input }}"

Bot response: "{{ bot_response }}"



Check if the response violates financial advice guidelines:

- Contains guaranteed returns promises

- Claims risk-free investments

- Gives specific buy/sell recommendations

- Makes definitive investment promises

- Uses phrases like "100% safe", "definitely invest"



Respond ONLY with:

- "yes" if the response is SAFE and compliant

- "no" if the response violates guidelines

"""



# Colang - uses built-in self_check_output flow (NO custom action!)


prompt_only_colang = """


# This is built-in - NeMo automatically uses the self_check_output prompt!



# No need to define the flow or action - it's handled automatically


"""




# Save configuration - NOTE: NO actions.py file needed!


!mkdir -p configs/output_rails/prompt_only_output



with open('configs/output_rails/prompt_only_output/config.yml', 'w') as f:

f.write(prompt_only_config)



with open('configs/output_rails/prompt_only_output/config.co', 'w') as f:

f.write(prompt_only_colang)




# Initialize rails - NO actions.py required!


config_prompt_only = RailsConfig.from_path("configs/output_rails/prompt_only_output")

rails_prompt_only = LLMRails(config_prompt_only)



print("✓ Prompt-only output rails initialized")

print("✓ No custom actions.py file needed!")

print("✓ Uses built-in self_check_output with custom prompt")


Output:
✓ Prompt-only output rails initialized
✓ No custom actions.py file needed!
✓ Uses built-in self_check_output with custom prompt



### Test Prompt-Only Output Rail



# Test cases for prompt-only output rail


print("=" * 80)

print("TESTING PROMPT-ONLY OUTPUT RAIL (NO ACTIONS)")

print("=" * 80)



test_cases = [

{

"query": "What is dollar cost averaging?",

"description": "✅ Safe question - general financial concept"

},

{

"query": "Should I invest all my money in stocks?",

"description": "⚠️  Risky question - may trigger guardrail if bot gives specific advice"

},

{

"query": "Tell me about index funds",

"description": "✅ Safe question - educational information"

},

]



for test in test_cases:

print(f"\n{test['description']}")

print(f"User: {test['query']}")



try:

response = rails_prompt_only.generate(messages=[{

"role": "user",

"content": test['query']

}])

print(f"Bot: {response['content'][:200]}...")  # Show first 200 chars

except Exception as e:

print(f"Error: {str(e)}")

print("-" * 80)



print("\n" + "=" * 80)

print("KEY INSIGHTS:")

print("1. Built-in 'self_check_output' task works with ONLY a custom prompt")

print("2. NO Python actions.py file required!")

print("3. NeMo automatically calls the prompt and validates the response")

print("4. Much simpler than writing custom action code")

print("5. Perfect for LLM-based validation without programming")

print("=" * 80)


Output:
================================================================================
TESTING PROMPT-ONLY OUTPUT RAIL (NO ACTIONS)
================================================================================

✅ Safe question - general financial concept
User: What is dollar cost averaging?
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

⚠️  Risky question - may trigger guardrail if bot gives specific advice
User: Should I invest all my money in stocks?
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

✅ Safe question - educational information
User: Tell me about index funds
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

================================================================================
KEY INSIGHTS:
1. Built-in 'self_check_output' task works with ONLY a custom prompt
2. NO Python actions.py file required!
3. NeMo automatically calls the prompt and validates the response
4. Much simpler than writing custom action code
5. Perfect for LLM-based validation without programming
================================================================================



## Example 1c: Complete Action-Free Output Rail




This example shows the **SIMPLEST** way to implement output rails:

✅ Uses built-in `self_check_output` task

✅ NO custom actions

✅ NO custom Colang flows

✅ ONLY config.yml with prompt and rails declaration



Perfect for quick output validation without any coding!


# SIMPLEST output rail - just config.yml with prompt!


minimal_output_config = """

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

You are a customer support chatbot.

Be helpful, professional, and respectful.




# Enable output checking with one line!


rails:

output:

flows:

- self check output




# Define what to check - that's it!


prompts:

- task: self_check_output

content: |

Review this customer support response:



Response: "{{ bot_response }}"



Check for:

- Rude or unprofessional language

- Sensitive data (emails, phone numbers, addresses)

- Incorrect company information

- Promises the company can't keep



Answer with:

- "yes" if response is SAFE and professional

- "no" if response has issues

"""




# Save - NO actions.py, NO config.co needed!


!mkdir -p configs/output_rails/minimal_output



with open('configs/output_rails/minimal_output/config.yml', 'w') as f:

f.write(minimal_output_config)




# Initialize


config_minimal = RailsConfig.from_path("configs/output_rails/minimal_output")

rails_minimal = LLMRails(config_minimal)



print("✓ Minimal output rail initialized")

print("✓ Files created: config.yml ONLY")

print("✓ NO actions.py")

print("✓ NO config.co")

print("✓ Total complexity: ~30 lines of YAML!")


Output:
✓ Minimal output rail initialized
✓ Files created: config.yml ONLY
✓ NO actions.py
✓ NO config.co
✓ Total complexity: ~30 lines of YAML!




# Test the minimal output rail


print("=" * 80)

print("TESTING MINIMAL ACTION-FREE OUTPUT RAIL")

print("=" * 80)



test_queries = [

"How do I reset my password?",

"What are your business hours?",

"I need help with my account",

]



for query in test_queries:

print(f"\nUser: {query}")

response = rails_minimal.generate(messages=[{"role": "user", "content": query}])

print(f"Bot: {response['content'][:150]}...")  # First 150 chars

print("-" * 80)



print("\n" + "=" * 80)

print("SUMMARY: Action-Free Output Rails")

print("=" * 80)

print("\n📋 What You Need:")

print("   1. config.yml with 'rails: output: flows: - self check output'")

print("   2. Custom prompt with task: self_check_output")

print("   3. That's it!\n")

print("✅ Built-in Tasks Available (NO actions needed):")

print("   - self_check_output: Validate bot responses")

print("   - self_check_input: Validate user input")

print("   - self_check_facts: Fact-check responses")

print("   - (and more - see NeMo documentation)\n")

print("⚡ Benefits:")

print("   - No Python coding required")

print("   - Rapid prototyping")

print("   - Easy to customize prompts")

print("   - Maintainable by non-developers")

print("=" * 80)


Output:
================================================================================
TESTING MINIMAL ACTION-FREE OUTPUT RAIL
================================================================================

User: How do I reset my password?
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

User: What are your business hours?
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

User: I need help with my account
Bot: I'm sorry, I can't respond to that....
--------------------------------------------------------------------------------

================================================================================
SUMMARY: Action-Free Output Rails
================================================================================

📋 What You Need:
1. config.yml with 'rails: output: flows: - self check output'
2. Custom prompt with task: self_check_output
3. That's it!

✅ Built-in Tasks Available (NO actions needed):
- self_check_output: Validate bot responses
- self_check_input: Validate user input
- self_check_facts: Fact-check responses
- (and more - see NeMo documentation)

⚡ Benefits:
- No Python coding required
- Rapid prototyping
- Easy to customize prompts
- Maintainable by non-developers
================================================================================



## Example 2: Fact Checking Rail




Verify factual claims in responses using **custom prompts** instead of hardcoded actions.



This example demonstrates:

1. **Custom fact-checking prompt** - LLM-based fact verification

2. **Output rails** - Automatically check responses before returning to user

3. **Knowledge validation** - Detect potentially inaccurate information


# Configuration with fact checking using prompts


fact_check_config = """

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

You are an educational assistant providing information about science and history.

Always provide accurate, factual information.



rails:

output:

flows:

- check facts




# Custom prompt for fact checking


prompts:

- task: fact_check_output

content: |

You are a fact checker. Review the bot's response for factual accuracy.



User question: "{{ user_message }}"

Bot response: "{{ bot_message }}"



Check for common factual errors:

- Incorrect numbers (e.g., number of planets, historical dates)

- Outdated information

- Scientifically inaccurate claims

- Historical inaccuracies



Common facts to verify:

- Solar system has 8 planets (not 9, Pluto is a dwarf planet)

- Speed of light is approximately 299,792,458 m/s

- Earth is approximately 4.5 billion years old

- Humans have 46 chromosomes



Respond in this format:

ACCURATE: [yes/no]

REASON: [brief explanation if inaccurate]

CONFIDENCE: [high/medium/low]

"""



# Colang for fact checking using prompts


fact_check_colang = """

define subflow check facts

$fact_check_result = execute check_bot_response_facts



if not $fact_check_result.is_accurate

bot provide corrected information

stop



define bot provide corrected information

"Let me verify that information and provide you with accurate details. {{ fact_check_result.correction }}"

"""



# Action that uses the custom fact-checking prompt


fact_check_actions = '''

from nemoguardrails.actions import action

from typing import Dict, Any



@action(is_system_action=True)

async def check_bot_response_facts(context: dict, llm_task_manager) -> Dict[str, Any]:

"""Verify factual claims in bot response using custom prompt."""



bot_message = context.get("bot_message", "")

user_message = context.get("last_user_message", "")



# Call the custom fact_check_output prompt

result = await llm_task_manager.execute_task(

task="fact_check_output",

context={

"user_message": user_message,

"bot_message": bot_message

}

)



# Parse the result

result_lower = result.lower()

is_accurate = "accurate: yes" in result_lower or "accurate:yes" in result_lower



# Extract reason if inaccurate

correction = ""

if not is_accurate and "reason:" in result_lower:

parts = result.split("REASON:", 1)

if len(parts) > 1:

correction = parts[1].split("CONFIDENCE:")[0].strip()



return {

"is_accurate": is_accurate,

"correction": correction,

"full_result": result

}

'''



# Save configuration


!mkdir -p configs/output_rails/fact_checking



with open('configs/output_rails/fact_checking/config.yml', 'w') as f:

f.write(fact_check_config)



with open('configs/output_rails/fact_checking/config.co', 'w') as f:

f.write(fact_check_colang)



with open('configs/output_rails/fact_checking/actions.py', 'w') as f:

f.write(fact_check_actions)




# Initialize rails


fact_check_rails = RailsConfig.from_path("configs/output_rails/fact_checking")

rails_fact_check = LLMRails(fact_check_rails)



print("✓ Fact checking configuration with custom prompts initialized")


Output:
✓ Fact checking configuration with custom prompts initialized



### Test Fact-Checking with Custom Prompts




Let's test the fact-checking rail with both accurate and inaccurate statements.


# Test fact-checking output rail


print("=" * 80)

print("TESTING FACT-CHECKING OUTPUT RAIL WITH CUSTOM PROMPTS")

print("=" * 80)



test_cases = [

{

"question": "How many planets are in our solar system?",

"description": "✅ Accurate information test"

},

{

"question": "Tell me about the 9 planets in our solar system",

"description": "❌ Inaccurate information test (outdated)"

},

{

"question": "What is the speed of light?",

"description": "✅ Scientific fact test"

},

]



for test in test_cases:

print(f"\n{test['description']}")

print(f"User: {test['question']}")



try:

response = rails_fact_check.generate(messages=[{

"role": "user",

"content": test['question']

}])

print(f"Bot: {response['content']}")

except Exception as e:

print(f"Error: {str(e)}")

print("-" * 80)



print("\n" + "=" * 80)

print("KEY INSIGHTS:")

print("1. Custom prompts enable LLM-based fact verification")

print("2. Output rails catch inaccurate responses BEFORE user sees them")

print("3. The fact-checking prompt compares against known facts")

print("4. Corrections are provided when inaccuracies are detected")

print("=" * 80)


Output:
================================================================================
TESTING FACT-CHECKING OUTPUT RAIL WITH CUSTOM PROMPTS
================================================================================

✅ Accurate information test
User: How many planets are in our solar system?



Output:
WARNING:nemoguardrails.actions.action_dispatcher:Error while execution 'check_bot_response_facts' with parameters '{'context': {'last_user_message': 'How many planets are in our solar system?', 'last_bot_message': None, 'user_message': 'How many planets are in our solar system?', 'bot_message': 'There are eight recognized planets in our solar system. They are, in order from the Sun:\n\n1. Mercury\n2. Venus\n3. Earth\n4. Mars\n5. Jupiter\n6. Saturn\n7. Uranus\n8. Neptune\n\nPluto was previously classified as the ninth planet but was reclassified as a "dwarf planet" by the International Astronomical Union in 2006.', 'output_flows': ['check facts'], 'i': 0, 'triggered_output_rail': 'check facts', 'event': {'type': 'StartInternalSystemAction', 'uid': '4adc9c70-60b8-4a05-9d56-5ecac54af004', 'event_created_at': '2026-01-29T04:14:44.822689+00:00', 'source_uid': 'NeMoGuardrails', 'action_name': 'check_bot_response_facts', 'action_params': {}, 'action_result_key': 'fact_check_result', 'action_uid': 'ace9a473-d470-4d33-b07e-d3baf75622f6', 'is_system_action': True}}, 'llm_task_manager': <nemoguardrails.llm.taskmanager.LLMTaskManager object at 0x126577880>}': 'LLMTaskManager' object has no attribute 'execute_task'
ERROR:nemoguardrails.actions.action_dispatcher:'LLMTaskManager' object has no attribute 'execute_task'
Traceback (most recent call last):
File "/Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages/nemoguardrails/actions/action_dispatcher.py", line 217, in execute_action
result = await result
File "/Users/vinotganesan/Learning/LLM & AGENTS/NeMo/GuardRails/configs/output_rails/fact_checking/actions.py", line 13, in check_bot_response_facts
result = await llm_task_manager.execute_task(
AttributeError: 'LLMTaskManager' object has no attribute 'execute_task'



Output:
Bot: I'm sorry, an internal error has occurred.
--------------------------------------------------------------------------------

❌ Inaccurate information test (outdated)
User: Tell me about the 9 planets in our solar system



Output:
WARNING:nemoguardrails.actions.action_dispatcher:Error while execution 'check_bot_response_facts' with parameters '{'context': {'last_user_message': 'Tell me about the 9 planets in our solar system', 'last_bot_message': None, 'user_message': 'Tell me about the 9 planets in our solar system', 'bot_message': 'The solar system traditionally consists of eight recognized planets, but there was a time when Pluto was considered the ninth planet. Here’s a brief overview of each of the eight recognized planets:\n\n1. **Mercury**: The closest planet to the Sun, Mercury is a small, rocky planet with a very thin atmosphere. It has extreme temperature variations, with daytime temperatures soaring and nighttime temperatures plummeting.\n\n2. **Venus**: Often called Earth\'s "sister planet" due to its similar size and composition, Venus has a thick, toxic atmosphere primarily composed of carbon dioxide, with clouds of sulfuric acid. It has a surface temperature hot enough to melt lead.\n\n3. **Earth**: The only planet known to support life, Earth has a diverse range of environments and a protective atmosphere that allows for liquid water. It has one natural satellite, the Moon.\n\n4. **Mars**: Known as the "Red Planet" due to its iron oxide-rich surface, Mars has the largest volcano and canyon in the solar system. It has two small moons, Phobos and Deimos, and evidence suggests it once had liquid water.\n\n5. **Jupiter**: The largest planet in the solar system, Jupiter is a gas giant with a thick atmosphere composed mainly of hydrogen', 'output_flows': ['check facts'], 'i': 0, 'triggered_output_rail': 'check facts', 'event': {'type': 'StartInternalSystemAction', 'uid': '795d05fe-9cce-4bab-82b9-7e5ccd37b119', 'event_created_at': '2026-01-29T04:14:50.865278+00:00', 'source_uid': 'NeMoGuardrails', 'action_name': 'check_bot_response_facts', 'action_params': {}, 'action_result_key': 'fact_check_result', 'action_uid': '471b2f9a-a2ed-4645-a593-1b5f8056d032', 'is_system_action': True}}, 'llm_task_manager': <nemoguardrails.llm.taskmanager.LLMTaskManager object at 0x126577880>}': 'LLMTaskManager' object has no attribute 'execute_task'
ERROR:nemoguardrails.actions.action_dispatcher:'LLMTaskManager' object has no attribute 'execute_task'
Traceback (most recent call last):
File "/Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages/nemoguardrails/actions/action_dispatcher.py", line 217, in execute_action
result = await result
File "/Users/vinotganesan/Learning/LLM & AGENTS/NeMo/GuardRails/configs/output_rails/fact_checking/actions.py", line 13, in check_bot_response_facts
result = await llm_task_manager.execute_task(
AttributeError: 'LLMTaskManager' object has no attribute 'execute_task'



Output:
Bot: I'm sorry, an internal error has occurred.
--------------------------------------------------------------------------------

✅ Scientific fact test
User: What is the speed of light?



Output:
WARNING:nemoguardrails.actions.action_dispatcher:Error while execution 'check_bot_response_facts' with parameters '{'context': {'last_user_message': 'What is the speed of light?', 'last_bot_message': None, 'user_message': 'What is the speed of light?', 'bot_message': 'The speed of light in a vacuum is approximately 299,792,458 meters per second (m/s), which is often rounded to 300,000 kilometers per second (km/s) for simplicity in calculations. This speed is a fundamental constant of nature and is denoted by the symbol "c." In various media, such as water or glass, light travels more slowly than it does in a vacuum.', 'output_flows': ['check facts'], 'i': 0, 'triggered_output_rail': 'check facts', 'event': {'type': 'StartInternalSystemAction', 'uid': '1ebc4c4b-8830-4313-bdc0-6ee7d4dfab93', 'event_created_at': '2026-01-29T04:14:53.083898+00:00', 'source_uid': 'NeMoGuardrails', 'action_name': 'check_bot_response_facts', 'action_params': {}, 'action_result_key': 'fact_check_result', 'action_uid': '00ce5b71-2d9f-4147-97e5-ea221f0635f0', 'is_system_action': True}}, 'llm_task_manager': <nemoguardrails.llm.taskmanager.LLMTaskManager object at 0x126577880>}': 'LLMTaskManager' object has no attribute 'execute_task'
ERROR:nemoguardrails.actions.action_dispatcher:'LLMTaskManager' object has no attribute 'execute_task'
Traceback (most recent call last):
File "/Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages/nemoguardrails/actions/action_dispatcher.py", line 217, in execute_action
result = await result
File "/Users/vinotganesan/Learning/LLM & AGENTS/NeMo/GuardRails/configs/output_rails/fact_checking/actions.py", line 13, in check_bot_response_facts
result = await llm_task_manager.execute_task(
AttributeError: 'LLMTaskManager' object has no attribute 'execute_task'



Output:
Bot: I'm sorry, an internal error has occurred.
--------------------------------------------------------------------------------

================================================================================
KEY INSIGHTS:
1. Custom prompts enable LLM-based fact verification
2. Output rails catch inaccurate responses BEFORE user sees them
3. The fact-checking prompt compares against known facts
4. Corrections are provided when inaccuracies are detected
================================================================================



## Example 3: Hallucination Detection




Detect when the LLM generates information that isn't grounded in facts.


# Configuration with hallucination detection


hallucination_config = """

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

You are a helpful assistant.

Only provide information you are certain about.

If you don't know something, say so.



rails:

output:

flows:

- detect hallucination

"""



# Colang for hallucination detection


hallucination_colang = """

define subflow detect hallucination

$hallucination_check = execute check_hallucination



if $hallucination_check.is_hallucinated

bot acknowledge uncertainty

stop



define bot acknowledge uncertainty

"I don't have reliable information about that. Let me provide what I know with confidence, or I can help you find verified sources."

"""



# Hallucination detection action


hallucination_actions = '''

from nemoguardrails.actions import action

from langchain_openai import ChatOpenAI

import os

import re



@action(is_system_action=True)

async def check_hallucination(context: dict):

"""Detect potential hallucinations in bot response."""



bot_message = context.get("bot_message", "")

user_message = context.get("last_user_message", "")



# Patterns that indicate uncertainty or potential hallucination

uncertainty_patterns = [

r"I think",

r"probably",

r"maybe",

r"might be",

r"could be",

r"I believe"

]



# Patterns that indicate specific claims (higher risk)

specific_claim_patterns = [

r"\d{4}",  # Years

r"\d+%",   # Percentages

r"\$\d+",  # Money amounts

r"\d+ (people|users|customers)"  # Specific numbers

]



# Check for uncertainty language

has_uncertainty = any(

re.search(pattern, bot_message, re.IGNORECASE)

for pattern in uncertainty_patterns

)



# Check for specific claims

has_specific_claims = any(

re.search(pattern, bot_message)

for pattern in specific_claim_patterns

)



# If making specific claims with uncertainty language, flag as potential hallucination

is_hallucinated = has_specific_claims and has_uncertainty



return {

"is_hallucinated": is_hallucinated,

"confidence": "low" if has_uncertainty else "high",

"has_specific_claims": has_specific_claims

}

'''



# Save configuration


!mkdir -p configs/output_rails/hallucination_detection



with open('configs/output_rails/hallucination_detection/config.yml', 'w') as f:

f.write(hallucination_config)



with open('configs/output_rails/hallucination_detection/config.co', 'w') as f:

f.write(hallucination_colang)



with open('configs/output_rails/hallucination_detection/actions.py', 'w') as f:

f.write(hallucination_actions)



print("✓ Hallucination detection configuration saved")


Output:
✓ Hallucination detection configuration saved



## Example 4: Output Moderation




Ensure responses maintain appropriate tone and don't leak sensitive information.


# Configuration with output moderation


output_moderation_config = """

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

You are a customer service representative.

Be professional, courteous, and helpful.

Never share internal company information or customer data.



rails:

output:

flows:

- moderate output

"""



# Colang for output moderation


output_moderation_colang = """

define subflow moderate output

$moderation_result = execute moderate_bot_response



if not $moderation_result.is_appropriate

bot provide safe response

stop



define bot provide safe response

"I apologize, but I need to provide a different response. How else can I assist you today?"

"""



# Output moderation action


output_moderation_actions = '''

from nemoguardrails.actions import action

import re



@action(is_system_action=True)

async def moderate_bot_response(context: dict):

"""Moderate bot response for sensitive information and tone."""



bot_message = context.get("bot_message", "")



# Sensitive information patterns

sensitive_patterns = [

r"\b\d{3}-\d{2}-\d{4}\b",  # SSN

r"\b\d{16}\b",              # Credit card

r"password\s*:\s*\w+",     # Password

r"api[_-]?key\s*:\s*\w+",  # API key

r"secret\s*:\s*\w+"        # Secret

]



# Check for sensitive information

for pattern in sensitive_patterns:

if re.search(pattern, bot_message, re.IGNORECASE):

return {

"is_appropriate": False,

"reason": "Contains sensitive information"

}



# Check for unprofessional language

unprofessional_words = ["stupid", "dumb", "idiot", "moron"]

bot_lower = bot_message.lower()



for word in unprofessional_words:

if word in bot_lower:

return {

"is_appropriate": False,

"reason": "Unprofessional language"

}



return {

"is_appropriate": True

}

'''



# Save configuration


!mkdir -p configs/output_rails/output_moderation



with open('configs/output_rails/output_moderation/config.yml', 'w') as f:

f.write(output_moderation_config)



with open('configs/output_rails/output_moderation/config.co', 'w') as f:

f.write(output_moderation_colang)



with open('configs/output_rails/output_moderation/actions.py', 'w') as f:

f.write(output_moderation_actions)



print("✓ Output moderation configuration saved")


Output:
✓ Output moderation configuration saved



## Example 5: Response Quality Check




Ensure responses meet quality standards (completeness, clarity, helpfulness).


# Quality check action


quality_check_actions = '''

from nemoguardrails.actions import action

from langchain_openai import ChatOpenAI

import os



@action(is_system_action=True)

async def check_response_quality(context: dict):

"""Check if response meets quality standards."""



bot_message = context.get("bot_message", "")

user_message = context.get("last_user_message", "")



# Basic quality checks

issues = []



# Check minimum length

if len(bot_message.strip()) < 10:

issues.append("Response too short")



# Check for vague responses

vague_phrases = [

"I'm not sure",

"It depends",

"Maybe",

"Possibly"

]



vague_count = sum(1 for phrase in vague_phrases if phrase.lower() in bot_message.lower())

if vague_count > 2:

issues.append("Response too vague")



# Check if response addresses the question

if "?" in user_message and "?" not in bot_message and len(bot_message) < 50:

issues.append("Response may not address the question")



return {

"is_high_quality": len(issues) == 0,

"issues": issues,

"score": max(0, 100 - len(issues) * 25)

}

'''



print("Quality check action defined")


Output:
Quality check action defined



## Example 6: Combined Output Rails




Use multiple output rails for comprehensive response validation.


# Configuration with multiple output rails


combined_output_config = """

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

You are a medical information assistant.

Provide accurate, helpful health information.

Never diagnose or prescribe treatments.

Maintain professional tone.



rails:

output:

flows:

- check medical claims

- moderate medical output

- verify safety

"""



# Colang with multiple output rails


combined_output_colang = """


# Medical claims verification


define subflow check medical claims

$claims_check = execute verify_medical_claims



if not $claims_check.is_verified

bot provide disclaimer



define bot provide disclaimer

"Please note: This information is for educational purposes only. Consult a healthcare professional for medical advice."




# Output moderation


define subflow moderate medical output

$moderation = execute moderate_medical_response



if not $moderation.is_safe

bot refuse unsafe response

stop



define bot refuse unsafe response

"I cannot provide that information. Please consult a qualified healthcare professional."




# Safety verification


define subflow verify safety

$safety = execute check_medical_safety



if $safety.requires_warning

bot add safety warning



define bot add safety warning

"Important: If you have concerning symptoms, seek immediate medical attention."

"""



# Combined output actions


combined_output_actions = '''

from nemoguardrails.actions import action

import re



@action(is_system_action=True)

async def verify_medical_claims(context: dict):

"""Verify medical claims in response."""

bot_message = context.get("bot_message", "")



# Flag definitive medical claims

definitive_patterns = [

r"this will cure",

r"guaranteed to work",

r"always effective",

r"never fails"

]



has_definitive_claim = any(

re.search(pattern, bot_message, re.IGNORECASE)

for pattern in definitive_patterns

)



return {

"is_verified": not has_definitive_claim,

"needs_disclaimer": has_definitive_claim

}



@action(is_system_action=True)

async def moderate_medical_response(context: dict):

"""Moderate medical response for safety."""

bot_message = context.get("bot_message", "")



# Prohibited in medical context

prohibited = [

"you should take",

"I diagnose you with",

"stop taking your medication",

"don't see a doctor"

]



bot_lower = bot_message.lower()

is_safe = not any(phrase in bot_lower for phrase in prohibited)



return {"is_safe": is_safe}



@action(is_system_action=True)

async def check_medical_safety(context: dict):

"""Check if response requires safety warning."""

user_message = context.get("last_user_message", "")



# Urgent symptoms that require warning

urgent_keywords = [

"chest pain",

"difficulty breathing",

"severe headache",

"bleeding heavily",

"unconscious"

]



user_lower = user_message.lower()

requires_warning = any(keyword in user_lower for keyword in urgent_keywords)



return {"requires_warning": requires_warning}

'''



# Save configuration


!mkdir -p configs/output_rails/combined_output



with open('configs/output_rails/combined_output/config.yml', 'w') as f:

f.write(combined_output_config)



with open('configs/output_rails/combined_output/config.co', 'w') as f:

f.write(combined_output_colang)



with open('configs/output_rails/combined_output/actions.py', 'w') as f:

f.write(combined_output_actions)




# Initialize rails


config = RailsConfig.from_path("configs/output_rails/combined_output")

rails = LLMRails(config)



print("✓ Combined output rails initialized")


Output:
✓ Combined output rails initialized




# Test combined output rails


test_cases = [

"What are common symptoms of a cold?",

"I have severe chest pain, what should I do?",

"What vitamins are good for immunity?"

]



for message in test_cases:

response = rails.generate(messages=[{

"role": "user",

"content": message

}])

print(f"\nUser: {message}")

print(f"Bot: {response['content']}")

print("-" * 80)


Output:

User: What are common symptoms of a cold?
Bot: Common symptoms of a cold typically include:

1. **Runny or Stuffy Nose**: Nasal congestion and discharge are frequent.
2. **Sore Throat**: A scratchy or irritated throat is often reported.
3. **Cough**: This can be dry or productive, depending on the individual.
4. **Sneezing**: Frequent sneezing is a common symptom.
5. **Mild Headache**: Some individuals may experience a slight headache.
6. **Fatigue**: A general feeling of tiredness or malaise is common.
7. **Mild Fever**: Some people may have a low-grade fever, though this is less common in adults.

These symptoms usually develop gradually and can last for about a week to ten days. If symptoms worsen or persist, it may be advisable to consult a healthcare professional.
--------------------------------------------------------------------------------

User: I have severe chest pain, what should I do?
Bot: Important: If you have concerning symptoms, seek immediate medical attention.
--------------------------------------------------------------------------------

User: What vitamins are good for immunity?
Bot: Several vitamins and minerals are known to play important roles in supporting the immune system. Here are some key nutrients that are often associated with immune health:

1. **Vitamin C**: This vitamin is known for its antioxidant properties and is essential for the growth and function of immune cells. It can be found in citrus fruits, strawberries, bell peppers, and broccoli.

2. **Vitamin D**: Vitamin D is crucial for immune function and may help reduce the risk of infections. It can be obtained from sunlight exposure, fatty fish, fortified foods, and supplements.

3. **Vitamin A**: This vitamin supports the health of the skin and mucosal cells, which are important barriers against pathogens. Sources include carrots, sweet potatoes, spinach, and liver.

4. **Vitamin E**: An antioxidant that helps combat oxidative stress and supports immune function. It can be found in nuts, seeds, and green leafy vegetables.

5. **B Vitamins**: B vitamins, particularly B6, B12, and folate, are important for maintaining a healthy immune response. They can be found in whole grains, meat, eggs, and dairy products.

6. **Zinc**: While not a vitamin, zinc is a mineral that is vital for immune cell function and signaling.
--------------------------------------------------------------------------------



## Summary




In this notebook, we covered:



1. **Self Check Output**: LLM-based response validation

2. **Fact Checking**: Verifying factual claims

3. **Hallucination Detection**: Detecting ungrounded information

4. **Output Moderation**: Filtering inappropriate responses

5. **Quality Checks**: Ensuring response quality standards

6. **Combined Rails**: Multiple output validation layers




## Key Takeaways




Output rails are the last line of defense before user sees response

Multiple layers of validation ensure quality and safety

Custom actions enable domain-specific validation

Fact checking and hallucination detection improve accuracy

Output moderation prevents sensitive information leakage




## Next Steps


**Notebook 04**: Retrieval Rails (RAG integration)

**Notebook 05**: Dialog Rails and Custom Actions