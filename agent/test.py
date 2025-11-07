import httpx
from dotenv import load_dotenv
import asyncio
import os
import json

load_dotenv()

async def chat():
    url = os.getenv("llm_key") 
    payload = {
        "model": os.getenv("model_name"),
        "messages": [
            {"role": "system", "content": "Esti un AI de test si specifica asta mereu"},
            {"role":"system","content":"Răspunde direct în limba română, NU include <think> sau explicații de raționament."},
        ],
        "temperature": 0.6,
        "max_tokens": 1024,
        "tool-chain": "none" 
    }

    print(f'Url = {url} \n payload = {payload} \n ')

    timeout = httpx.Timeout(connect=1000.0, read=120000.0, write=100000.0, pool=500000000.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            payload["messages"].append(
                {"role": "user", "content": user_input}
            )
            
            res = await client.post(url=url, json=payload)
            

            message = res.json()["choices"][0]["message"]
            if "content" in message:
                print("Model: ", message["content"])
            else:
                print("Eroare")

if __name__ == "__main__":
    asyncio.run(chat())