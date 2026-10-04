import json
import os
from openai import OpenAI

# Initialize client (Ensure OPENAI_API_KEY is set in your terminal environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))

# The 17 prototype signs supported by your live baseline model
KNOWN_SIGNS = [
    "bank", "biglarge", "black", "blue", "cellphone", "good", "hello",
    "hot", "new", "pen", "red", "shoes", "smalllittle", "storeorshop",
    "thankyou", "tshirt", "white"
]

def parse_to_semantic_sequence(text: str) -> list:
    """
    Stage 2 & 3: Converts unstructured speech into a dynamic ISL concept array.
    """
    prompt = f"""
    You are an ISL (Indian Sign Language) semantic sequence planner.
    Convert the user's speech (English, Hindi, or Hinglish) into an ordered list of concepts.

    RULES:
    1. You MUST ONLY output concepts present in this exact list: {KNOWN_SIGNS}
    2. Map synonyms (e.g., "shirt" -> "tshirt", "namaste" -> "hello", "bada" -> "biglarge", "mobile" -> "cellphone").
    3. Ignore conversational grammar, filler words, and unsupported words.
    4. Output strictly as a valid JSON array of strings, nothing else.

    User Speech: "{text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_json = response.choices[0].message.content.strip()
        sequence = json.loads(raw_json)
        
        # Filter to valid signs
        validated = [sign for sign in sequence if sign in KNOWN_SIGNS]
        return validated if validated else ["hello"]
        
    except Exception as e:
        print(f"[NLU Error] Fallback triggered: {e}")
        return ["hello"]