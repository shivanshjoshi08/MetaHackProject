import os
import json
from openai import OpenAI
from environment import EmailTriageEnv
from tasks import grade_task1, grade_task2, grade_task3

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", # Groq ka OpenAI-compatible endpoint
    api_key=os.environ.get("GROQ_API_KEY", "your_api_key_here") 
)

# Groq par Llama 3.1 ka fast version
MODEL = "llama-3.1-8b-instant"
GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}

SYSTEM_PROMPT = """
You are an AI agent acting as a customer support specialist.
You will receive an inbox of support emails and must process them.
Always respond with a single JSON action object. Available actions:
{"action_type": "categorize", "email_id": "...", "category": "..."}
{"action_type": "query_db", "customer_email": "..."}
{"action_type": "draft", "email_id": "...", "draft_text": "..."}
{"action_type": "send", "email_id": "..."}

Categories: Billing | Technical | Sales | Spam | Other
For Billing or Order Status emails: query the database first,
then include the customer's Order ID in your draft response.
"""

def build_prompt(obs) -> str:
    return f"Current step: {obs.step_number}. Inbox has {len(obs.inbox)} emails. What is your next action?"

def parse_action(raw_dict: dict):
    # Dummy parser to dispatch to Pydantic models based on action_type
    from environment import CategorizeEmail, QueryDatabase, DraftResponse, SendResponse
    act_type = raw_dict.get("action_type")
    if act_type == "categorize": return CategorizeEmail(**raw_dict)
    if act_type == "query_db": return QueryDatabase(**raw_dict)
    if act_type == "draft": return DraftResponse(**raw_dict)
    if act_type == "send": return SendResponse(**raw_dict)
    raise ValueError("Unknown action type")

def run_task(task_id: int) -> float:
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)
    env._max_steps = {1: 3, 2: 15, 3: 40}[task_id]
    done = False
    
    while not done:
        prompt = build_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.0
            )
            raw = json.loads(response.choices[0].message.content)
            action = parse_action(raw)
        except Exception as e:
            print(f"Inference/Parsing failed: {e}")
            break
            
        obs, reward, done, info = env.step(action)
        print(f"Step {obs.step_number}: reward={reward.score:.2f} | {reward.reason}")
        
    return GRADERS[task_id](env.state())

if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        score = run_task(task_id)
        print(f"Task {task_id} Final Score: {score:.4f}")