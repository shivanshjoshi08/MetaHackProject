"""
Baseline Inference Script for OpenEnv Email Triage
===================================================
Provides both:
  1. An async generator `run_task_stream(task_id)` used by the FastAPI WebSocket
  2. A standalone CLI runner via `python baseline_inference.py`

All LLM calls go through Groq's OpenAI-compatible endpoint.
The API key is read from the GROQ_API_KEY environment variable.
"""

import os
import json
import asyncio
from openai import OpenAI
from environment import (
    EmailTriageEnv,
    CategorizeEmail,
    QueryDatabase,
    DraftResponse,
    SendResponse,
)
from tasks import grade_task1, grade_task2, grade_task3

# ─── LLM Client ────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "your_api_key_here"),
)
MODEL = "llama-3.1-8b-instant"
GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}

SYSTEM_PROMPT = """
You are an AI agent acting as a customer support specialist.
You will receive an inbox of support emails and must process them.

Always respond with a SINGLE JSON action object (no extra text). Available actions:
  {"action_type": "categorize", "email_id": "...", "category": "..."}
  {"action_type": "query_db", "customer_email": "..."}
  {"action_type": "draft", "email_id": "...", "draft_text": "..."}
  {"action_type": "send", "email_id": "..."}

Categories: Billing | Technical | Sales | Spam | Other

STRATEGY:
1. First, categorize every email in the inbox one by one.
2. For Billing emails: after categorizing, query the database using the sender's email.
3. After getting DB results, draft a response that includes the customer's Order ID.
4. After drafting, send the response.
5. Continue until all emails are processed.

Process emails in order (email_001 first, then email_002, etc.).
"""


# def build_prompt(obs) -> str:
#     """Build a rich prompt that gives the LLM full situational awareness."""
#     lines = [
#         f"Task ID: {obs.task_id} | Step: {obs.step_number}",
#         f"Inbox emails ({len(obs.inbox)} total):",
#     ]
#     for email in obs.inbox:
#         lines.append(f"  - {email.id}: from={email.sender} subject=\"{email.subject}\"")

#     lines.append(f"\nAlready processed (categorized): {obs.processed}")

#     if obs.db_query_result:
#         lines.append(f"\nLast DB query result: {json.dumps(obs.db_query_result)}")
#     else:
#         lines.append("\nNo DB query result available.")

#     if obs.drafted_responses:
#         lines.append(f"Drafted responses exist for: {list(obs.drafted_responses.keys())}")

#     uncategorized = [e.id for e in obs.inbox if e.id not in obs.processed]
#     if uncategorized:
#         lines.append(f"\nEmails still needing categorization: {uncategorized}")
#         lines.append(f"Next email to process: {uncategorized[0]}")
#     else:
#         lines.append("\nAll emails are categorized. Draft/send responses if needed, or the task may be done.")

#     lines.append("\nWhat is your next SINGLE action? Reply with only a JSON object.")
#     return "\n".join(lines)
def build_prompt(obs) -> str:
    # Un emails ko filter karo jo abhi tak categorize nahi hue hain
    unprocessed_emails = [e for e in obs.inbox if e.id not in obs.processed]
    
    prompt = f"Current step: {obs.step_number}.\n"
    prompt += f"Already categorized emails: {obs.processed}\n"
    prompt += f"Drafts ready for: {list(obs.drafted_responses.keys())}\n\n"
    
    if obs.db_query_result:
        prompt += f"Last DB Query Result: {obs.db_query_result}\n\n"
        
    prompt += "--- INBOX DETAILS ---\n"
    for e in obs.inbox:
        status = "PROCESSED" if e.id in obs.processed else "NEEDS CATEGORIZATION"
        prompt += f"[{status}] ID: {e.id} | Sender: {e.sender} | Subject: {e.subject} | Body: {e.body}\n"
        
    prompt += "\nINSTRUCTIONS:\n"
    prompt += "1. If an email is 'NEEDS CATEGORIZATION', categorize it.\n"
    prompt += "2. If it is Billing/Order related, query_db using their sender email.\n"
    prompt += "3. Draft a response using the Order ID from the DB.\n"
    prompt += "4. Do NOT repeat an action you have already taken!\n"
    prompt += "What is your single NEXT action?"
    
    return prompt


def parse_action(raw_dict: dict):
    """Dispatch a raw dict into the correct Pydantic action model."""
    act_type = raw_dict.get("action_type")
    if act_type == "categorize":
        return CategorizeEmail(**raw_dict)
    if act_type == "query_db":
        return QueryDatabase(**raw_dict)
    if act_type == "draft":
        return DraftResponse(**raw_dict)
    if act_type == "send":
        return SendResponse(**raw_dict)
    raise ValueError(f"Unknown action type: {act_type}")


# ─── Async Generator (used by WebSocket) ───────────────────────────────────────

async def run_task_stream(task_id: int):
    """
    Async generator that yields JSON-serialisable event dicts.
    Each yield is one WebSocket message to the frontend.
    """
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)
    max_steps = {1: 3, 2: 15, 3: 40}[task_id]
    env._max_steps = max_steps
    difficulty = {1: "Easy", 2: "Medium", 3: "Hard"}[task_id]

    # ── Event: INIT ──
    inbox_data = []
    for email in obs.inbox:
        inbox_data.append({
            "id": email.id,
            "sender": email.sender,
            "subject": email.subject,
            "snippet": email.body[:60] + ("..." if len(email.body) > 60 else ""),
        })

    yield {
        "event": "init",
        "type": "info",
        "text": f"🔄 Initializing Task {task_id} ({difficulty}) — {len(obs.inbox)} emails loaded",
        "task_id": task_id,
        "max_steps": max_steps,
        "step": 0,
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "inbox": inbox_data,
    }
    await asyncio.sleep(0.5)

    done = False
    step_count = 0

    while not done:
        # ── Event: THINKING ──
        yield {
            "event": "thinking",
            "type": "info",
            "text": f"🧠 Agent analysing state (step {obs.step_number + 1})...",
            "step": obs.step_number,
            "max_steps": max_steps,
            "cumulative_reward": env._cumulative_reward,
        }
        await asyncio.sleep(0.3)

        # ── Call LLM ──
        prompt = build_prompt(obs)
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.0,
            )
            raw = json.loads(response.choices[0].message.content)
            action = parse_action(raw)
        except Exception as e:
            yield {
                "event": "error",
                "type": "penalty",
                "text": f"❌ Inference or parsing failed: {str(e)}",
                "step": obs.step_number,
                "max_steps": max_steps,
                "cumulative_reward": env._cumulative_reward,
            }
            break

        # ── Event: ACTION PREVIEW ──
        action_text = ""
        event_type = "action"
        category_data = None
        db_data = None

        if action.action_type == "categorize":
            event_type = "action"
            action_text = f'🏷️  Categorizing {action.email_id} → "{action.category}"'
        elif action.action_type == "query_db":
            event_type = "db"
            action_text = f"🔍 Querying database for {action.customer_email}..."
        elif action.action_type == "draft":
            event_type = "draft"
            preview = action.draft_text[:80] + ("..." if len(action.draft_text) > 80 else "")
            action_text = f'✍️  Drafting response for {action.email_id}: "{preview}"'
        elif action.action_type == "send":
            event_type = "action"
            action_text = f"📤 Sending response for {action.email_id}"

        # ── Execute step ──
        obs, reward, done, info = env.step(action)
        step_count = obs.step_number

        # Build category data if it was a categorize action
        if action.action_type == "categorize":
            category_data = {"id": action.email_id, "cat": action.category}

        # Build DB result data if the query returned something
        if action.action_type == "query_db" and obs.db_query_result:
            db_data = obs.db_query_result

        yield {
            "event": "step",
            "type": event_type,
            "text": action_text,
            "step": obs.step_number,
            "max_steps": max_steps,
            "reward": reward.score,
            "cumulative_reward": reward.cumulative,
            "reason": reward.reason,
            "penalty": reward.penalty_applied,
            "category": category_data,
            "db_result": db_data,
        }
        await asyncio.sleep(0.4)

        # ── Reward feedback event ──
        if reward.score != 0:
            r_icon = "✅" if reward.score > 0 else "⚠️"
            yield {
                "event": "reward",
                "type": "reward" if reward.score > 0 else "penalty",
                "text": f"{r_icon} {reward.reason} ({'+' if reward.score > 0 else ''}{reward.score:.2f})",
                "step": obs.step_number,
                "max_steps": max_steps,
                "reward": reward.score,
                "cumulative_reward": reward.cumulative,
            }
            await asyncio.sleep(0.2)

        # ── DB result follow-up event ──
        if db_data:
            name = db_data.get("name", "Unknown")
            oid = db_data.get("order_id", "N/A")
            status = db_data.get("order_status", "N/A")
            billing = db_data.get("billing_issue", "None")
            yield {
                "event": "db_result_detail",
                "type": "dbresult",
                "text": f"📋 DB Hit! {name} | {oid} | Status: {status} | Issue: {billing}",
                "step": obs.step_number,
                "max_steps": max_steps,
                "cumulative_reward": reward.cumulative,
                "db_result": db_data,
            }
            await asyncio.sleep(0.3)

    # ── Event: COMPLETE ──
    final_score = GRADERS[task_id](env.state())
    yield {
        "event": "complete",
        "type": "success",
        "text": f"🏁 Task {task_id} Complete — Final Score: {final_score:.4f}",
        "step": step_count,
        "max_steps": max_steps,
        "cumulative_reward": env._cumulative_reward,
        "final_score": final_score,
        "done": True,
    }


# ─── Synchronous CLI Runner ────────────────────────────────────────────────────

def run_task(task_id: int) -> float:
    """Original synchronous runner for standalone CLI use."""
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)
    env._max_steps = {1: 3, 2: 15, 3: 40}[task_id]
    done = False

    print(f"\n{'='*50}")
    print(f"  Task {task_id} — Starting")
    print(f"{'='*50}")

    while not done:
        prompt = build_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.0,
            )
            raw = json.loads(response.choices[0].message.content)
            action = parse_action(raw)
        except Exception as e:
            print(f"  ❌ Inference/Parsing failed: {e}")
            break

        obs, reward, done, info = env.step(action)
        sign = "+" if reward.score >= 0 else ""
        print(f"  Step {obs.step_number}: {sign}{reward.score:.2f} | {reward.reason}")

    final = GRADERS[task_id](env.state())
    print(f"  Final Score: {final:.4f}\n")
    return final


if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        score = run_task(task_id)
        print(f"Task {task_id} Final Score: {score:.4f}")