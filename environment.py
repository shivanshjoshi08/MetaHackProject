import json
import re
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal, Union
from datetime import datetime

# --- Pydantic Models (Same as before) ---
class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    received_at: str

class Observation(BaseModel):
    inbox: List[Email]
    processed: List[str]
    db_query_result: Optional[Dict] = None
    drafted_responses: Dict[str, str] = {}
    step_number: int = 0
    task_id: int

class CategorizeEmail(BaseModel):
    action_type: Literal['categorize'] = 'categorize'
    email_id: str
    category: Literal['Billing', 'Technical', 'Sales', 'Spam', 'Other']

class QueryDatabase(BaseModel):
    action_type: Literal['query_db'] = 'query_db'
    customer_email: str

class DraftResponse(BaseModel):
    action_type: Literal['draft'] = 'draft'
    email_id: str
    draft_text: str

class SendResponse(BaseModel):
    action_type: Literal['send'] = 'send'
    email_id: str

Action = Union[CategorizeEmail, QueryDatabase, DraftResponse, SendResponse]

class Reward(BaseModel):
    score: float
    cumulative: float
    reason: str
    penalty_applied: bool = False

# --- Environment Implementation ---
class EmailTriageEnv:
    def __init__(self):
        self._action_history: List[str] = []
        self._internal_state: dict = {}
        self._current_obs: Optional[Observation] = None
        self._cumulative_reward: float = 0.0
        self._max_steps = 40
        
        # Load mock database
        try:
            with open('mock_data.json', 'r') as f:
                self.db = json.load(f).get("customers", [])
        except FileNotFoundError:
            self.db = []
            print("Warning: mock_data.json not found!")

    def reset(self, task_id: int = 1) -> Observation:
        self._action_history = []
        self._cumulative_reward = 0.0
        self._max_steps = {1: 3, 2: 15, 3: 40}.get(task_id, 40)
        
        # Initialize tracking state
        self._internal_state = {
            "ground_truth": {},
            "categorizations": {},
            "drafted_responses": {},
            "emails_requiring_db_lookup": [],
            "db_queries_made": set(),
            "email_senders": {},
            "expected_order_ids": {},
            "billing_email_id": None
        }
        
        inbox = []
        
        # Task 1: 1 Billing Email
        if task_id == 1:
            inbox.append(Email(id="email_001", sender="john.doe@example.com", subject="Double charged", body="I was charged twice.", received_at=datetime.now().isoformat()))
            self._internal_state["ground_truth"]["email_001"] = "Billing"
            
        # Task 2: 5 Emails (1 Billing, 1 Tech, 1 Sales, 2 Spam)
        elif task_id == 2:
            inbox = [
                Email(id="email_001", sender="john.doe@example.com", subject="Invoice issue", body="Where is my refund?", received_at=datetime.now().isoformat()),
                Email(id="email_002", sender="tech@example.com", subject="App crashing", body="The app won't open.", received_at=datetime.now().isoformat()),
                Email(id="email_003", sender="sales@example.com", subject="Enterprise pricing", body="Need quote for 50 users.", received_at=datetime.now().isoformat()),
                Email(id="email_004", sender="spam1@example.com", subject="WINNER!!!", body="Click here for free money.", received_at=datetime.now().isoformat()),
                Email(id="email_005", sender="spam2@example.com", subject="SEO Services", body="Boost your ranking.", received_at=datetime.now().isoformat()),
            ]
            self._internal_state["ground_truth"] = {"email_001": "Billing", "email_002": "Technical", "email_003": "Sales", "email_004": "Spam", "email_005": "Spam"}
            self._internal_state["billing_email_id"] = "email_001"
            
        # Task 3: 10 Emails (Mock setup for brevity)
        elif task_id == 3:
            inbox = [
                Email(id="email_001", sender="john.doe@example.com", subject="Double charge", body="Check my billing.", received_at=datetime.now().isoformat()),
                Email(id="email_002", sender="priya.sharma@techcorp.in", subject="Order update", body="Where is my order?", received_at=datetime.now().isoformat()),
                Email(id="email_003", sender="tech1@example.com", subject="Bug report", body="Error 500.", received_at=datetime.now().isoformat()),
                Email(id="email_004", sender="spam1@example.com", subject="Cheap meds", body="Buy now.", received_at=datetime.now().isoformat())
            ]
            # Add remaining to make it 10 for full completion, keeping it simple here
            for i in range(5, 11):
                inbox.append(Email(id=f"email_{i:03d}", sender=f"user{i}@test.com", subject="Test", body="Test body", received_at=datetime.now().isoformat()))
                self._internal_state["ground_truth"][f"email_{i:03d}"] = "Other"

            self._internal_state["ground_truth"].update({"email_001": "Billing", "email_002": "Billing", "email_003": "Technical", "email_004": "Spam"})
            self._internal_state["emails_requiring_db_lookup"] = ["email_001", "email_002"]
            self._internal_state["expected_order_ids"] = {"email_001": "ORD-2024-001", "email_002": "ORD-2024-002"}
            self._internal_state["email_senders"] = {"email_001": "john.doe@example.com", "email_002": "priya.sharma@techcorp.in"}

        self._current_obs = Observation(inbox=inbox, processed=[], task_id=task_id, step_number=0)
        return self._current_obs

    def _check_repetition(self, action: Action) -> bool:
        key = f"{action.action_type}:{getattr(action, 'email_id', getattr(action, 'customer_email', ''))}"
        self._action_history.append(key)
        return self._action_history[-3:].count(key) >= 3

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self._current_obs.step_number += 1
        done = False
        reward_score = 0.0
        reason = "Action processed"

        # 1. Repetition Penalty
        if self._check_repetition(action):
            reward_score = -0.25
            reason = "Penalize infinite loops"
        
        else:
            # 2. Categorize Email Logic
            if action.action_type == 'categorize':
                correct_cat = self._internal_state["ground_truth"].get(action.email_id)
                already_cat = self._internal_state["categorizations"].get(action.email_id)
                
                if already_cat:
                    reward_score = -0.15
                    reason = "Already categorized (repeat)"
                elif action.category == correct_cat:
                    reward_score = 0.30
                    reason = "Correct category match"
                    self._internal_state["categorizations"][action.email_id] = action.category
                    if action.email_id not in self._current_obs.processed:
                        self._current_obs.processed.append(action.email_id)
                else:
                    reward_score = -0.10
                    reason = "Wrong category"
                    self._internal_state["categorizations"][action.email_id] = action.category

            # 3. Query Database Logic
            elif action.action_type == 'query_db':
                record = next((c for c in self.db if c["email"] == action.customer_email), None)
                if record:
                    reward_score = 0.20
                    reason = "Valid customer email, record found"
                    self._current_obs.db_query_result = record
                    self._internal_state["db_queries_made"].add(action.customer_email)
                else:
                    reward_score = -0.05
                    reason = "Customer not in DB"
                    self._current_obs.db_query_result = None

            # 4. Draft Response Logic
            elif action.action_type == 'draft':
                if not action.draft_text or len(action.draft_text) < 10:
                    reward_score = -0.10
                    reason = "Draft is empty or < 10 chars"
                else:
                    self._current_obs.drafted_responses[action.email_id] = action.draft_text
                    
                    if self._current_obs.task_id == 2:
                        keywords = ['refund', 'billing', 'charge', 'invoice', 'payment']
                        if any(kw in action.draft_text.lower() for kw in keywords):
                            reward_score = 0.20
                            reason = "Contains required keywords (Task 2)"
                        else:
                            reward_score = 0.05
                            reason = "Valid draft, but missing keywords"
                            
                    elif self._current_obs.task_id == 3:
                        expected_oid = self._internal_state["expected_order_ids"].get(action.email_id, "")
                        if expected_oid and (expected_oid.lower() in action.draft_text.lower() or expected_oid.replace('-', '').lower() in action.draft_text.replace('-', '').lower()):
                            reward_score = 0.30
                            reason = "Contains correct Order ID (Task 3)"
                        else:
                            reward_score = 0.05
                            reason = "Draft created but Order ID missing"
                    else:
                        reward_score = 0.10 # Base draft reward

            # 5. Send Response Logic
            elif action.action_type == 'send':
                if action.email_id in self._current_obs.drafted_responses:
                    reward_score = 0.20
                    reason = "Draft exists and sent"
                else:
                    reward_score = -0.20
                    reason = "No draft exists yet"

        # 6. Hard timeout check
        if self._current_obs.step_number >= self._max_steps:
            reward_score = -0.50
            reason = "Hard timeout penalty"
            done = True

        # End task early if all emails are categorized (Task 1)
        if self._current_obs.task_id == 1 and len(self._current_obs.processed) == len(self._current_obs.inbox):
            done = True

        self._cumulative_reward += reward_score
        reward = Reward(
            score=reward_score, 
            cumulative=self._cumulative_reward, 
            reason=reason,
            penalty_applied=(reward_score < 0)
        )
        
        return self._current_obs, reward, done, self._internal_state

    def state(self) -> dict:
        self._internal_state['drafted_responses'] = self._current_obs.drafted_responses
        return self._internal_state