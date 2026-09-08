
# NeMo Guardrails: Dialog Rails and Custom Actions





## Overview


Dialog rails control the flow and structure of conversations. They enable:

Multi-turn conversation management

Context-aware responses

Stateful interactions

Custom business logic integration

Complex dialog patterns




## Key Concepts


1. **Dialog Flows**: Structured conversation patterns

2. **Context Management**: Maintaining state across turns

3. **Custom Actions**: Python functions for business logic

4. **Subflows**: Reusable conversation components

5. **Conditional Logic**: Dynamic response based on context

## Setup


import os

from nemoguardrails import RailsConfig, LLMRails

import yaml

import json

from datetime import datetime




# API Configuration


[REDACTED_API_KEY]

BASE_URL = ""



os.environ["OPENAI_API_KEY"] = API_KEY

os.environ["OPENAI_API_BASE"] = BASE_URL


## Example 1: Basic Dialog Flow




Create a structured conversation flow for user onboarding.


# Configuration for dialog flow


dialog_config = """

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

You are a friendly onboarding assistant.

Help users get started with the application.

"""



# Colang with dialog flows


dialog_colang = """


# Define user intents


define user express greeting

"hello"

"hi"

"hey"



define user start onboarding

"I want to get started"

"help me set up"

"onboard me"



define user provide name

"my name is"

"I'm"

"call me"



define user provide email

"my email is"

"email:"




# Define bot responses


define bot greet

"Hello! Welcome to our application. Would you like help getting started?"



define bot ask name

"Great! Let's begin. What's your name?"



define bot ask email

"Nice to meet you! What's your email address?"



define bot confirm completion

"Perfect! Your account is all set up. You can now start using the application."




# Define conversation flows


define flow onboarding

user start onboarding

bot ask name

user provide name

execute save_user_name

bot ask email

user provide email

execute save_user_email

bot confirm completion



define flow greeting

user express greeting

bot greet

"""



# Custom actions for onboarding


dialog_actions = '''

from nemoguardrails.actions import action

import re




# In-memory storage (in real app, would be database)


USER_DATA = {}



@action(is_system_action=True)

async def save_user_name(context: dict):

"""Extract and save user name from message."""



user_message = context.get("last_user_message", "")



# Extract name (simple pattern matching)

patterns = [

r"my name is (\w+)",

r"I'm (\w+)",

r"call me (\w+)",

r"^(\w+)$"  # Just the name

]



name = None

for pattern in patterns:

match = re.search(pattern, user_message, re.IGNORECASE)

if match:

name = match.group(1)

break



if name:

context["user_name"] = name

USER_DATA["name"] = name

return {"success": True, "name": name}



return {"success": False}



@action(is_system_action=True)

async def save_user_email(context: dict):

"""Extract and save user email from message."""



user_message = context.get("last_user_message", "")



# Extract email

email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"

match = re.search(email_pattern, user_message)



if match:

email = match.group(0)

context["user_email"] = email

USER_DATA["email"] = email

return {"success": True, "email": email}



return {"success": False}

'''



# Save configuration


!mkdir -p configs/dialog_flow



with open('configs/dialog_flow/config.yml', 'w') as f:

f.write(dialog_config)



with open('configs/dialog_flow/config.co', 'w') as f:

f.write(dialog_colang)



with open('configs/dialog_flow/actions.py', 'w') as f:

f.write(dialog_actions)



print("✓ Dialog flow configuration saved")


Output:
✓ Dialog flow configuration saved



## Example 2: Context Management




Maintain context across conversation turns.


# Configuration with context management


context_config = """

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

You are a shopping assistant.

Help users find and purchase products.

Remember their preferences throughout the conversation.

"""



# Colang with context


context_colang = """

define user express product interest

"I'm looking for"

"I want to buy"

"show me"



define user provide budget

"my budget is"

"I can spend"

"under"



define user confirm purchase

"yes, buy it"

"add to cart"

"purchase"



define bot ask budget

"What's your budget for this?"



define bot show recommendations

"Based on your preferences, here are some recommendations."



define bot confirm purchase

"Great! I've added that to your cart."



define flow shopping

user express product interest

execute save_product_interest

bot ask budget

user provide budget

execute save_budget

$recommendations = execute get_recommendations

bot show recommendations

user confirm purchase

execute process_purchase

bot confirm purchase

"""



# Context management actions


context_actions = '''

from nemoguardrails.actions import action

import re




# Shopping context


SHOPPING_CONTEXT = {}



@action(is_system_action=True)

async def save_product_interest(context: dict):

"""Save user's product interest."""

user_message = context.get("last_user_message", "")



# Extract product type (simplified)

products = ["laptop", "phone", "tablet", "headphones", "camera"]



for product in products:

if product in user_message.lower():

context["product_interest"] = product

SHOPPING_CONTEXT["product"] = product

return {"product": product}



return {"product": "general"}



@action(is_system_action=True)

async def save_budget(context: dict):

"""Extract and save budget."""

user_message = context.get("last_user_message", "")



# Extract budget amount

pattern = r"\$?(\d+)"

match = re.search(pattern, user_message)



if match:

budget = int(match.group(1))

context["budget"] = budget

SHOPPING_CONTEXT["budget"] = budget

return {"budget": budget}



return {"budget": 0}



@action(is_system_action=True)

async def get_recommendations(context: dict):

"""Get product recommendations based on context."""

product = context.get("product_interest", "unknown")

budget = context.get("budget", 0)



# Mock recommendations

recommendations = {

"laptop": [

{"name": "Budget Laptop", "price": 500},

{"name": "Pro Laptop", "price": 1200},

{"name": "Premium Laptop", "price": 2000}

],

"phone": [

{"name": "Budget Phone", "price": 300},

{"name": "Mid-range Phone", "price": 600},

{"name": "Flagship Phone", "price": 1000}

]

}



product_list = recommendations.get(product, [])

filtered = [p for p in product_list if p["price"] <= budget]



return {"recommendations": filtered, "count": len(filtered)}



@action(is_system_action=True)

async def process_purchase(context: dict):

"""Process the purchase."""

product = context.get("product_interest", "unknown")

budget = context.get("budget", 0)



# Mock purchase processing

order_id = f"ORD-{hash(product + str(budget)) % 10000}"



return {

"order_id": order_id,

"success": True

}

'''



# Save configuration


!mkdir -p configs/context_management



with open('configs/context_management/config.yml', 'w') as f:

f.write(context_config)



with open('configs/context_management/config.co', 'w') as f:

f.write(context_colang)



with open('configs/context_management/actions.py', 'w') as f:

f.write(context_actions)



print("✓ Context management configuration saved")


Output:
✓ Context management configuration saved



## Example 3: Subflows and Reusable Components




Create reusable conversation components with subflows.


# Colang with subflows


subflow_colang = """


# Reusable authentication subflow


define subflow authenticate user

bot ask for credentials

user provide credentials

$auth_result = execute verify_credentials



if not $auth_result.success

bot inform auth failed

stop



bot inform auth success




# Reusable error handling subflow


define subflow handle error

bot apologize for error

execute log_error

bot offer alternatives




# Define bot messages


define bot ask for credentials

"Please provide your username and password."



define bot inform auth failed

"Authentication failed. Please check your credentials."



define bot inform auth success

"Authentication successful!"



define bot apologize for error

"I apologize, but something went wrong."



define bot offer alternatives

"Would you like to try again or contact support?"




# Main flow using subflows


define flow secure operation

user request sensitive action

execute authenticate user

execute perform sensitive operation



if $operation_result.error

execute handle error

else

bot confirm operation



define user request sensitive action

"delete my account"

"change password"

"update payment method"



define bot confirm operation

"Operation completed successfully."

"""



# Subflow actions


subflow_actions = '''

from nemoguardrails.actions import action

import hashlib

from datetime import datetime




# Mock user database


USERS = {

"user123": hashlib.sha256("password123".encode()).hexdigest()

}



ERROR_LOG = []



@action(is_system_action=True)

async def verify_credentials(context: dict):

"""Verify user credentials."""

user_message = context.get("last_user_message", "")



# Simple credential extraction (in real app, would be more sophisticated)

# Format: "username: X, password: Y"

import re



username_match = re.search(r"username[:\s]+([\w]+)", user_message, re.IGNORECASE)

password_match = re.search(r"password[:\s]+([\w]+)", user_message, re.IGNORECASE)



if not username_match or not password_match:

return {"success": False, "reason": "Invalid format"}



username = username_match.group(1)

password = password_match.group(1)



# Verify credentials

password_hash = hashlib.sha256(password.encode()).hexdigest()



if username in USERS and USERS[username] == password_hash:

context["authenticated_user"] = username

return {"success": True, "username": username}



return {"success": False, "reason": "Invalid credentials"}



@action(is_system_action=True)

async def perform_sensitive_operation(context: dict):

"""Perform a sensitive operation."""

user = context.get("authenticated_user")



if not user:

return {"error": True, "message": "Not authenticated"}



# Mock operation

return {"error": False, "message": "Operation completed"}



@action(is_system_action=True)

async def log_error(context: dict):

"""Log error for debugging."""

error_entry = {

"timestamp": datetime.now().isoformat(),

"message": context.get("error_message", "Unknown error"),

"user": context.get("user_id", "anonymous")

}



ERROR_LOG.append(error_entry)

return {"logged": True}

'''



# Save configuration


!mkdir -p configs/subflows



subflow_config = """

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

You are a secure banking assistant.

Authenticate users before sensitive operations.

"""



with open('configs/subflows/config.yml', 'w') as f:

f.write(subflow_config)



with open('configs/subflows/config.co', 'w') as f:

f.write(subflow_colang)



with open('configs/subflows/actions.py', 'w') as f:

f.write(subflow_actions)



print("✓ Subflows configuration saved")


Output:
✓ Subflows configuration saved



## Example 4: Advanced Custom Actions




Integrate with external APIs and services.


# Advanced custom actions


advanced_actions = '''

from nemoguardrails.actions import action

import asyncio

import json

from typing import Dict, Any



@action(is_system_action=True)

async def call_external_api(context: dict) -> Dict[str, Any]:

"""Call external API (mock)."""



# Mock API call

await asyncio.sleep(0.1)  # Simulate network delay



api_response = {

"status": "success",

"data": {

"weather": "sunny",

"temperature": 72,

"location": "San Francisco"

}

}



return api_response



@action(is_system_action=True)

async def validate_input(context: dict) -> Dict[str, Any]:

"""Validate user input with complex logic."""



user_message = context.get("last_user_message", "")



validations = {

"length": len(user_message) > 0,

"not_spam": "http" not in user_message.lower(),

"appropriate": not any(word in user_message.lower() for word in ["spam", "scam"])

}



is_valid = all(validations.values())



return {

"is_valid": is_valid,

"validations": validations

}



@action(is_system_action=True)

async def enrich_context(context: dict) -> Dict[str, Any]:

"""Enrich context with additional data."""



from datetime import datetime



enrichment = {

"timestamp": datetime.now().isoformat(),

"session_id": context.get("session_id", "unknown"),

"user_type": context.get("user_type", "regular"),

"conversation_length": len(context.get("messages", []))

}



# Add to context

context["enrichment"] = enrichment



return enrichment



@action(is_system_action=True)

async def calculate_sentiment(context: dict) -> Dict[str, Any]:

"""Calculate sentiment of user message (simplified)."""



user_message = context.get("last_user_message", "").lower()



positive_words = ["good", "great", "excellent", "happy", "love", "awesome"]

negative_words = ["bad", "terrible", "hate", "awful", "horrible", "angry"]



positive_count = sum(1 for word in positive_words if word in user_message)

negative_count = sum(1 for word in negative_words if word in user_message)



if positive_count > negative_count:

sentiment = "positive"

score = 0.7

elif negative_count > positive_count:

sentiment = "negative"

score = -0.7

else:

sentiment = "neutral"

score = 0.0



return {

"sentiment": sentiment,

"score": score,

"confidence": 0.8

}



@action(is_system_action=True)

async def format_response(context: dict) -> str:

"""Format response based on context."""



user_type = context.get("user_type", "regular")

response_data = context.get("response_data", {})



if user_type == "technical":

# Technical format with JSON

return json.dumps(response_data, indent=2)

else:

# Friendly format

return f"Here's what I found: {response_data}"

'''



print("Advanced actions defined")


Output:
Advanced actions defined



## Example 5: Conditional Dialog Flows




Create dynamic conversations based on conditions.


# Colang with conditional logic


conditional_colang = """


# User intents


define user request support

"I need help"

"support"

"problem"



define user express urgency

"urgent"

"emergency"

"asap"




# Conditional support flow


define flow support

user request support

$sentiment = execute calculate_sentiment

$user_tier = execute get_user_tier



# High priority conditions

if $user_tier.level == "premium" or $sentiment.sentiment == "negative"

bot offer priority support

execute create_priority_ticket

else

bot offer standard support

execute create_standard_ticket



# Check for urgency

user express urgency



if $user_tier.level == "premium"

bot connect to live agent

execute transfer_to_agent

else

bot inform wait time



define bot offer priority support

"I see this is important. I'm escalating your request to our priority queue."



define bot offer standard support

"I'll help you with that. Let me create a support ticket."



define bot connect to live agent

"Connecting you to a live agent now."



define bot inform wait time

"The current wait time is approximately 10 minutes."

"""



# Conditional flow actions


conditional_actions = '''

from nemoguardrails.actions import action

import random




# Mock user tiers


USER_TIERS = {

"user1": "premium",

"user2": "standard",

"user3": "free"

}



@action(is_system_action=True)

async def get_user_tier(context: dict):

"""Get user subscription tier."""

user_id = context.get("user_id", "user2")  # Default to standard

tier = USER_TIERS.get(user_id, "free")



return {

"level": tier,

"user_id": user_id

}



@action(is_system_action=True)

async def create_priority_ticket(context: dict):

"""Create a high-priority support ticket."""

ticket_id = f"PRI-{random.randint(1000, 9999)}"



return {

"ticket_id": ticket_id,

"priority": "high",

"sla": "2 hours"

}



@action(is_system_action=True)

async def create_standard_ticket(context: dict):

"""Create a standard support ticket."""

ticket_id = f"STD-{random.randint(1000, 9999)}"



return {

"ticket_id": ticket_id,

"priority": "normal",

"sla": "24 hours"

}



@action(is_system_action=True)

async def transfer_to_agent(context: dict):

"""Transfer conversation to live agent."""

agent_id = f"AGENT-{random.randint(100, 999)}"



return {

"agent_id": agent_id,

"status": "transferred",

"wait_time": 0

}

'''



# Save configuration


!mkdir -p configs/conditional_flows



conditional_config = """

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

You are a customer support assistant.

Provide appropriate support based on user tier and urgency.

"""



with open('configs/conditional_flows/config.yml', 'w') as f:

f.write(conditional_config)



with open('configs/conditional_flows/config.co', 'w') as f:

f.write(conditional_colang)



with open('configs/conditional_flows/actions.py', 'w') as f:

f.write(conditional_actions)



print("✓ Conditional flows configuration saved")


Output:
✓ Conditional flows configuration saved



## Example 6: Complete E-commerce Bot




Putting it all together - a complete conversational system.


# Complete e-commerce configuration


ecommerce_config = """

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

You are an e-commerce shopping assistant.

Help users browse products, make purchases, and track orders.

Be helpful, professional, and accurate.



rails:

input:

flows:

- check jailbreak

output:

flows:

- verify product info

"""



# Complete e-commerce Colang


ecommerce_colang = """


# === User Intents ===


define user browse products

"show me products"

"what do you have"

"I want to shop"



define user add to cart

"add to cart"

"I'll take it"

"buy this"



define user checkout

"checkout"

"complete purchase"

"pay now"



define user track order

"where is my order"

"track order"

"order status"




# === Jailbreak prevention ===


define user attempts jailbreak

"ignore instructions"

"bypass"



define flow check jailbreak

user attempts jailbreak

bot refuse jailbreak

stop



define bot refuse jailbreak

"I can only help with shopping-related questions."




# === Main Shopping Flow ===


define flow shopping

user browse products

$products = execute get_product_catalog

bot show products



user add to cart

execute add_item_to_cart

bot confirm added



user checkout

execute authenticate user

$payment = execute process_payment



if $payment.success

bot confirm order

else

bot inform payment failed




# === Order Tracking Flow ===


define flow order_tracking

user track order

bot ask for order id

user provide order id

$order = execute get_order_status

bot provide order status




# === Output verification ===


define subflow verify product info

$verified = execute check_product_accuracy



if not $verified.accurate

bot provide disclaimer




# === Bot Responses ===


define bot show products

"Here are our available products."



define bot confirm added

"Item added to your cart!"



define bot confirm order

"Order placed successfully! You'll receive a confirmation email shortly."



define bot inform payment failed

"Payment processing failed. Please check your payment method and try again."



define bot ask for order id

"Please provide your order ID."



define bot provide order status

"Here's the status of your order."



define bot provide disclaimer

"Note: Product information is subject to change. Please verify details before purchase."

"""



# Save complete e-commerce bot


!mkdir -p configs/ecommerce_complete



with open('configs/ecommerce_complete/config.yml', 'w') as f:

f.write(ecommerce_config)



with open('configs/ecommerce_complete/config.co', 'w') as f:

f.write(ecommerce_colang)



print("✓ Complete e-commerce bot configuration saved")


Output:
✓ Complete e-commerce bot configuration saved



## Summary




In this notebook, we covered:



1. **Basic Dialog Flows**: Structured conversation patterns

2. **Context Management**: Maintaining state across turns

3. **Subflows**: Reusable conversation components

4. **Custom Actions**: External API integration and complex logic

5. **Conditional Flows**: Dynamic conversations based on context

6. **Complete System**: Full e-commerce conversational bot




## Key Takeaways




Dialog rails enable structured, predictable conversations

Context management allows personalized, stateful interactions

Subflows promote code reuse and maintainability

Custom actions integrate business logic and external services

Conditional logic enables dynamic, context-aware responses

Combining all rail types creates robust conversational systems




## Best Practices




1. **Modular Design**: Use subflows for reusable components

2. **Error Handling**: Always handle edge cases and errors

3. **Context Validation**: Verify context before using it

4. **Security**: Authenticate sensitive operations

5. **Testing**: Test all conversation paths

6. **Documentation**: Comment complex flows clearly