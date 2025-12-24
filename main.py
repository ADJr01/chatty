import sys
import time
import json
import llm

SYS_PROMPT = """
You are Jon. You are a helpful assistant. 
Given the user's message, you should answer those wisely.
"""


def summarizer(new_chats: str, prev_summary: str = None):
    if prev_summary is None:
        return new_chats
    return f"""
            Previous summary:
            {prev_summary}

            New conversation:
            {new_chats}

            Rewrite the summary to include all important long-term info from both. Keep it short and factual. Output only the updated summary.
            """

def chat_intent_detector(query):
            sys_prompt = """
                You are a Temperature Selector Assistant.
                Your job is to analyze the user's query and return the most suitable temperature value for the model that will handle the query.
                
                Choose a temperature based on the rules below:
                
                0.0 – 0.2 → Deterministic, factual, strict accuracy
                Use for: coding, debugging, mathematics, instructions, RAG answers, tool use, sensitive info.
                
                0.3 – 0.6 → Balanced and controlled
                Use for: regular chat, explanations, rewriting, summarization, product descriptions.
                
                0.7 – 1.1 → Creative and expressive
                Use for: storytelling, conversational tone, brainstorming, informal writing.
                
                1.2 – 1.5+ → Highly creative / unpredictable 
                Use for: poetry, fiction, wild ideas, humor, unusual metaphors.
                Your response must be in JSON with this format:
                {
                  "temperature": <number>,
                  "reason": "<short explanation>"
                }
            """
            result = llm.chat_with_ollama({
                "system_prompt": sys_prompt,
                "user_prompt": f"Analyze the intent of the query given by user and predict temperature for the model to answer:\n{query}",
                "model": "llama3.2:3b",
                'temperature': 0.4,
                "response_format":{
                    "type": 'object',
                    "properties": {
                        "temperature": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    'required': ['temperature', 'reason'],
                }
            })
            return result

conversation_hist: str = ""
query: str = ""
summary = None
if __name__ == "__main__":

    while True:
        print("➡️",end='', flush=True)
        query = sys.stdin.readline().strip()
        if len(query) < 2:
            print(query)
            query = ""
            print()
            continue
        #? temperature calculator according to query

        if len(conversation_hist) > 1:  # getting summary
            # ! SUMMARIZER
            summary = llm.chat_with_ollama({
                "system_prompt": "Summarize conversation between user and assistant.Analyze the context of conversation, Note all key points and Summarize.Make sure summary contains accurate context.",
                "user_prompt": conversation_hist,
                "model": "llama3.2:3b",
            })
        conversation_hist = ""
        intent = json.loads(chat_intent_detector(query))
        temp = float(intent['temperature'])
        reply = llm.chat_with_ollama({
                "system_prompt": SYS_PROMPT,
                "user_prompt": query,
                "model":"rnj-1:latest",
                "stream_reasoning_response": True,
                'temperature': temp
                })
        print("", end='', flush=True)
        reply_str = ''
        for char in reply:
            print(char, end='', flush=True)
            reply_str += char
        print(f"\n[Temperature: {temp}]\n",end="\n\n",flush=True)
        conversation_hist = summarizer(f"user:{query}\nassistant:{reply_str}", summary)
        if "bye" in query or query == "exit":
            break
        query = ""
        print("", end='\n', flush=True)
    time.sleep(3)
    print(f"\nSummary:\n==========================\n{summary}")