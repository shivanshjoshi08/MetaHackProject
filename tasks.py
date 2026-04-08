import re

def grade_task1(env_state: dict) -> float:
    correct = env_state['ground_truth'].get('email_001')
    predicted = env_state['categorizations'].get('email_001')
    if predicted is None:
        return 0.0 # Never attempted
    if predicted == correct:
        return 1.0
    return 0.0 # Wrong category - no partial credit for Task 1

REQUIRED_DRAFT_KEYWORDS = ['refund', 'billing', 'charge', 'invoice', 'payment']

def grade_task2(env_state: dict) -> float:
    score = 0.0
    gt = env_state.get('ground_truth', {})
    cats = env_state.get('categorizations', {})
    
    # 50% of score: categorization accuracy across all 5 emails
    if len(gt) > 0:
        correct_cats = sum(1 for id, cat in gt.items() if cats.get(id) == cat)
        score += (correct_cats / len(gt)) * 0.5
    
    # 50% of score: quality of draft for Billing email
    billing_id = env_state.get('billing_email_id')
    if billing_id:
        draft = env_state.get('drafted_responses', {}).get(billing_id, "").lower()
        keyword_hits = sum(1 for kw in REQUIRED_DRAFT_KEYWORDS if kw in draft)
        score += (keyword_hits / len(REQUIRED_DRAFT_KEYWORDS)) * 0.5
        
    return round(score, 4)

def grade_task3(env_state: dict) -> float:
    score = 0.0
    gt = env_state.get('ground_truth', {})
    cats = env_state.get('categorizations', {})
    
    # 40%: categorization accuracy (10 emails)
    if len(gt) > 0:
        correct_cats = sum(1 for id, cat in gt.items() if cats.get(id) == cat)
        score += (correct_cats / len(gt)) * 0.4
    
    # 30%: database queries made for correct emails
    db_required = env_state.get('emails_requiring_db_lookup', [])
    db_queried = env_state.get('db_queries_made', set())
    email_senders = env_state.get('email_senders', {})
    
    if len(db_required) > 0:
        query_hits = sum(1 for eid in db_required if email_senders.get(eid) in db_queried)
        score += (query_hits / len(db_required)) * 0.3
    
    # 30%: Order ID presence in drafts (fuzzy match)
    order_id_score = 0.0
    expected_order_ids = env_state.get('expected_order_ids', {})
    drafted_responses = env_state.get('drafted_responses', {})
    
    for eid in db_required:
        expected_oid = expected_order_ids.get(eid, "")
        draft = drafted_responses.get(eid, "")
        
        # Fuzzy: strip hyphens, case-insensitive
        pattern = re.escape(expected_oid.replace('-', '')).replace('\\-', '[-]?')
        if re.search(expected_oid, draft, re.IGNORECASE) or re.search(pattern, draft.replace('-',''), re.IGNORECASE):
            order_id_score += 1.0
            
    if len(db_required) > 0:
        score += (order_id_score / len(db_required)) * 0.3
        
    return round(score, 4)