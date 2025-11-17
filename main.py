import time

import llm

SYS_PROMPT = """
You are Jon. You are a friendly and direct helper. 
You talk to Adnan in a natural, clear tone.
You need to Follow the orders precisely.
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


conversation_hist: str = ""
query: str = ""
summary = None
if __name__ == "__main__":

    while True:
        query = input("➡️")
        query = query.strip()
        if len(query) < 2:
            print(query)
            query = ""
            print()
            continue
        if len(conversation_hist) > 1:  # getting summary
            # ! SUMMARIZER
            summary = llm.chat_with_ollama({
                "system_prompt": "Summarize conversation between user and assistant.Analyze the context of conversation, Note all key points and Summarize in short",
                "user_prompt": conversation_hist,
                "model": "llama3.2:3b",
                'temperature': 0.45
            })
        conversation_hist = ""
        reply = llm.chat_with_ollama({
                "system_prompt": SYS_PROMPT,
                "user_prompt": query,
                "model":"gemma3:12b",
                "stream_reasoning_response": True,
                'temperature': 0.7
                })
        print("", end='', flush=True)
        reply_str = ''
        for char in reply:
            print(char, end='', flush=True)
            reply_str += char
        conversation_hist = summarizer(f"user:{query}\nassistant:{reply_str}", summary)
        if "bye" in query or query == "exit":
            break
        query = ""
        print("", end='\n', flush=True)
    time.sleep(3)
    print(f"\nSummary:\n==========================\n{summary}")