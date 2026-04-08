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
You will receive a support email inbox and must process every email completely.

ALWAYS reply with exactly ONE JSON object. No explanation, no markdown, no extra text.

Available action types:
  {"action_type": "categorize", "email_id": "email_001", "category": "Billing"}
  {"action_type": "query_db", "customer_email": "user@example.com"}
  {"action_type": "draft", "email_id": "email_001", "draft_text": "Dear customer..."}
  {"action_type": "send", "email_id": "email_001"}

VALID CATEGORIES (use EXACTLY one of these strings, nothing else):
  Billing | Technical | Sales | Spam | Other

STEP-BY-STEP STRATEGY:
  Phase 1 — CATEGORIZE: For each email NOT YET CATEGORIZED, output a 'categorize' action.
  Phase 2 — DB LOOKUP: For each Billing/Order email, output a 'query_db' action using that sender's email.
  Phase 3 — DRAFT: For each Billing email, output a 'draft' action. Include the Order ID from DB results.
  Phase 4 — SEND: For each drafted email, output a 'send' action.

RULES:
  - NEVER use label text like 'NEEDS CATEGORIZATION' as a category value.
  - NEVER repeat an action for an email that is already done.
  - Look at the current state carefully and pick the NEXT logical action.

  If you have already drafted a response for an email, do NOT draft it again. Move to the next email or use the SendResponse action to conclude the task.
"""


def build_prompt(obs) -> str:
    """Build a state-aware prompt that never leaks label text as category values."""
    uncategorized = [e for e in obs.inbox if e.id not in obs.processed]
    billing_ids = [
        eid for eid, cat in (
            # Only check emails that have been categorized
            {e.id: obs.processed for e in obs.inbox if e.id in obs.processed}
        ).items()
    ]

    lines = [
        f"=== CURRENT STATE (Step {obs.step_number}) ===",
        f"Task: {obs.task_id}",
        "",
        "--- INBOX ---",
    ]

    for e in obs.inbox:
        done = e.id in obs.processed
        drafted = e.id in obs.drafted_responses
        tag = "[CATEGORIZED]" if done else "[NOT CATEGORIZED]"
        dtag = " [DRAFT READY]" if drafted else ""
        lines.append(
            f"  {tag}{dtag} {e.id} | from: {e.sender} | subject: {e.subject} | body: {e.body}"
        )

    lines.append("")
    lines.append(f"Categorized email IDs so far: {obs.processed}")
    lines.append(f"Draft responses exist for: {list(obs.drafted_responses.keys())}")

    if obs.db_query_result:
        lines.append(f"\nLast DB result: {json.dumps(obs.db_query_result)}")
    else:
        lines.append("\nNo DB query has been made yet.")

    lines.append("")
    lines.append("--- WHAT TO DO NEXT ---")

    if uncategorized:
        next_email = uncategorized[0]
        lines.append(f"Action needed: CATEGORIZE {next_email.id} (from {next_email.sender}, subject: {next_email.subject})")
        lines.append("Choose EXACTLY one of these category strings: Billing, Technical, Sales, Spam, Other")
        lines.append(f'Output: {{"action_type": "categorize", "email_id": "{next_email.id}", "category": "<ONE OF THE 5 VALID CATEGORIES>"}}')        
    else:
        lines.append("All emails are categorized. Move to Phase 2/3/4:")
        lines.append("  - If a Billing email has no DB lookup yet: use query_db with the sender email.")
        lines.append("  - If DB result exists and no draft yet: use draft (include the Order ID in draft_text).")
        lines.append("  - If draft exists and not sent yet: use send.")

    lines.append("\nReply with ONE JSON object only. No extra text.")
    return "\n".join(lines)


# Mapping to catch common LLM mislabellings before Pydantic validation
_CATEGORY_NORMALISE = {
    "billing": "Billing",
    "technical": "Technical",
    "tech": "Technical",
    "sales": "Sales",
    "spam": "Spam",
    "other": "Other",
}

def parse_action(raw_dict: dict):
    """Dispatch a raw dict into the correct Pydantic action model.
    
    Normalises the category field before validation so that case differences
    or minor LLM typos don't cause hard crashes.
    """
    act_type = raw_dict.get("action_type")

    if act_type == "categorize":
        # Normalise category to a valid value
        raw_cat = str(raw_dict.get("category", "")).strip()
        normalised = _CATEGORY_NORMALISE.get(raw_cat.lower())
        if normalised:
            raw_dict = {**raw_dict, "category": normalised}
        elif raw_cat not in ("Billing", "Technical", "Sales", "Spam", "Other"):
            # Last resort: reject unknown values with a clear message
            raise ValueError(
                f"LLM returned invalid category '{raw_cat}'. "
                f"Valid values: Billing, Technical, Sales, Spam, Other"
            )
        return CategorizeEmail(**raw_dict)

    if act_type == "query_db":
        return QueryDatabase(**raw_dict)
    if act_type == "draft":
        return DraftResponse(**raw_dict)
    if act_type == "send":
        return SendResponse(**raw_dict)
    raise ValueError(f"Unknown action type: '{act_type}'")


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

        # ── Call LLM (with up to 2 retries on parse failure) ──
        prompt = build_prompt(obs)
        action = None
        last_error = None
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):  # up to 3 attempts per step
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL,
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                    temperature=0.0,
                )
                raw = json.loads(response.choices[0].message.content)
                action = parse_action(raw)
                break  # success
            except Exception as e:
                last_error = e
                # Feed the error back as an assistant + user pair for the next attempt
                retry_messages = retry_messages + [
                    {"role": "assistant", "content": response.choices[0].message.content
                        if 'response' in dir() and response.choices else "{}"},
                    {"role": "user", "content": (
                        f"Your previous response caused a validation error: {e}\n"
                        "Reply ONLY with a corrected JSON object. "
                        "For category, use EXACTLY one of: Billing, Technical, Sales, Spam, Other"
                    )},
                ]
                await asyncio.sleep(0.2)

        if action is None:
            yield {
                "event": "error",
                "type": "penalty",
                "text": f"❌ Agent failed after 3 attempts: {last_error}",
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