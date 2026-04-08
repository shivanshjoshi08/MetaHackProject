---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Customer Support Email Triage Environment

This is an **OpenEnv-compliant reinforcement learning environment** centered on Customer Support Email Triage — a real-world task. It requires an agent to read, categorize, query a database, and draft appropriate responses for support emails.

## Environment Details
- **Tasks Included**:
  1. **Single Email Categorization (Easy)**
  2. **Inbox Triage + Draft Response (Medium)**
  3. **Full Triage + DB Lookup + Personalized Response (Hard)**
- **Observation Space**: Pydantic Structured Observation
- **Action Space**: Union [CategorizeEmail | QueryDatabase | DraftResponse | SendResponse]
- **Reward**: Dense (Each sub-step yields + / - rewards).

## Evaluation
A deterministic Baseline Agent script using `baseline_inference.py` is included. You can validate and test natively with standard OpenEnv tools:
```bash
openenv validate openenv.yaml