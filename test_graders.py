import pytest
from tasks import grade_task1, grade_task2, grade_task3

def test_task1_correct():
    state = {
        'ground_truth': {'email_001': 'Billing'},
        'categorizations': {'email_001': 'Billing'}
    }
    assert grade_task1(state) == 1.0

def test_task1_wrong():
    state = {
        'ground_truth': {'email_001': 'Billing'},
        'categorizations': {'email_001': 'Technical'}
    }
    assert grade_task1(state) == 0.0

def test_task2_full_score():
    state = {
        'ground_truth': {
            'email_001': 'Billing', 'email_002': 'Technical',
            'email_003': 'Sales', 'email_004': 'Spam', 'email_005': 'Spam'
        },
        'categorizations': {
            'email_001': 'Billing', 'email_002': 'Technical',
            'email_003': 'Sales', 'email_004': 'Spam', 'email_005': 'Spam'
        },
        'billing_email_id': 'email_001',
        'drafted_responses': {
            'email_001': 'we will process your refund for the charge associated with this invoice and billing payment.'
        }
    }
    assert grade_task2(state) == 1.0

def test_task3_fuzzy_order_id():
    state = {
        'ground_truth': {'email_001': 'Billing'},
        'categorizations': {'email_001': 'Billing'},
        'emails_requiring_db_lookup': ['email_001'],
        'db_queries_made': {'john.doe@example.com'},
        'email_senders': {'email_001': 'john.doe@example.com'},
        'expected_order_ids': {'email_001': 'ORD-2024-001'},
        'drafted_responses': {'email_001': 'Here is your ORD2024001 status'}
    }
    # 40% (1/1 cat) + 30% (1/1 query) + 30% (1/1 draft fuzzy match) = 1.0
    assert grade_task3(state) == 1.0
