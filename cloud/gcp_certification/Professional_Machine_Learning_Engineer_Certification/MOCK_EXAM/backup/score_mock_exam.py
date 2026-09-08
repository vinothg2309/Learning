#!/usr/bin/env python3
"""
GCP PMLE Mock Exam Scoring Script

Usage:
    1. Take the test in MOCK_Q_A.md
    2. Mark your answers by changing [ ] to [x]
    3. Save as MOCK_Q_A_COMPLETED.md
    4. Run: python3 score_mock_exam.py
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_answers(content):
    """Extract all checked answers from markdown file."""
    # Pattern: - [x] A. or - [x] B. etc.
    pattern = r'### Q(\d+)\..*?\n.*?(?:\n- \[x\] ([A-D])\.|(?!- \[x\]))'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

    answers = {}
    for match in matches:
        q_num = int(match.group(1))
        # Find all [x] marks for this question (for multi-select)
        q_text = match.group(0)
        checked = re.findall(r'- \[x\] ([A-D])\.', q_text)
        if checked:
            answers[q_num] = set(checked)

    return answers

def extract_answers_simple(content):
    """Extract answers with simpler pattern matching."""
    answers = {}
    lines = content.split('\n')
    current_q = None
    q_answers = set()

    for line in lines:
        # Match question header
        q_match = re.match(r'### Q(\d+)\.', line)
        if q_match:
            # Save previous question if any
            if current_q is not None and q_answers:
                answers[current_q] = q_answers
            current_q = int(q_match.group(1))
            q_answers = set()

        # Match checked answer
        if current_q is not None and re.match(r'- \[x\] ([A-D])\.', line):
            checked = re.match(r'- \[x\] ([A-D])\.', line)
            if checked:
                q_answers.add(checked.group(1))

    # Save last question
    if current_q is not None and q_answers:
        answers[current_q] = q_answers

    return answers

def score_exam(completed_file, answer_key_file):
    """Compare completed exam with answer key and score."""

    # Check files exist
    if not Path(completed_file).exists():
        print(f"❌ Error: {completed_file} not found")
        print(f"   Did you save your answers as {completed_file}?")
        return False

    if not Path(answer_key_file).exists():
        print(f"❌ Error: {answer_key_file} not found")
        return False

    # Read files
    with open(completed_file, 'r', encoding='utf-8') as f:
        your_content = f.read()

    with open(answer_key_file, 'r', encoding='utf-8') as f:
        key_content = f.read()

    # Extract answers
    your_answers = extract_answers_simple(your_content)
    correct_answers = extract_answers_simple(key_content)

    if not your_answers:
        print("❌ No answers found in your file. Did you mark any answers with [x]?")
        return False

    if not correct_answers:
        print("❌ No answers found in answer key file.")
        return False

    # Score
    correct_count = 0
    domain_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for q_num in sorted(correct_answers.keys()):
        correct_set = correct_answers[q_num]
        your_set = your_answers.get(q_num, set())

        # Determine domain
        if q_num <= 20:
            domain = "BigQuery ML"
        elif q_num <= 40:
            domain = "Vertex AI Pipelines"
        elif q_num <= 60:
            domain = "Data Engineering/Dataflow"
        elif q_num <= 80:
            domain = "Feature Store"
        elif q_num <= 100:
            domain = "Distributed Training"
        elif q_num <= 115:
            domain = "Model Evaluation"
        elif q_num <= 140:
            domain = "MLOps & CI/CD"
        elif q_num <= 155:
            domain = "Model Monitoring"
        elif q_num <= 175:
            domain = "Deployment & Serving"
        elif q_num <= 185:
            domain = "Hyperparameter Tuning"
        elif q_num <= 210:
            domain = "GenAI & RAG"
        elif q_num <= 230:
            domain = "Responsible AI"
        else:
            domain = "Production ML Systems"

        # For multi-select, exact match required
        is_correct = your_set == correct_set

        if is_correct:
            correct_count += 1

        domain_scores[domain]["total"] += 1
        if is_correct:
            domain_scores[domain]["correct"] += 1

    # Print results
    total = len(correct_answers)
    percentage = (correct_count / total * 100) if total > 0 else 0
    passing_score = int(total * 0.70)

    print("\n" + "="*60)
    print("GCP PMLE MOCK EXAM SCORE REPORT")
    print("="*60)
    print(f"\nOverall Score: {correct_count}/{total} ({percentage:.1f}%)")
    print(f"Passing Score (70%): {passing_score}/{total}")

    if percentage >= 70:
        print("\n✅ PASS - You're ready for the exam!")
    elif percentage >= 50:
        print("\n⚠️  NEEDS REVIEW - Study weak domains and retake")
    else:
        print("\n❌ FAIL - Intensive review recommended")

    # Domain breakdown
    print("\n" + "-"*60)
    print("Score by Domain:")
    print("-"*60)

    for domain in sorted(domain_scores.keys()):
        scores = domain_scores[domain]
        domain_pct = (scores["correct"] / scores["total"] * 100) if scores["total"] > 0 else 0
        status = "✓" if domain_pct >= 70 else "✗"
        print(f"{status} {domain:.<40} {scores['correct']}/{scores['total']} ({domain_pct:>5.1f}%)")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)

    if percentage >= 70:
        print("• Review explanations for incorrect answers")
        print("• Take another practice test in 2-3 days")
        print("• Focus on edge cases and trade-offs")
        print("• You're ready to schedule the real exam!")
    else:
        weak_domains = [d for d, s in domain_scores.items() if s["correct"] / s["total"] < 0.70]
        print(f"• Focus on weak domains: {', '.join(weak_domains)}")
        print("• Review MASTER_FILE.md sections for these domains")
        print("• Retake this test after 1 week of study")

    print("\n" + "="*60 + "\n")
    return True

if __name__ == '__main__':
    # Default paths (in same directory as script)
    script_dir = Path(__file__).parent
    completed = script_dir / 'MOCK_Q_A_COMPLETED.md'
    answer_key = script_dir / 'MOCK_Q_A_ANSWERS.md'

    # Allow command line arguments
    if len(sys.argv) > 1:
        completed = Path(sys.argv[1])
    if len(sys.argv) > 2:
        answer_key = Path(sys.argv[2])

    print(f"\nScoring exam...")
    print(f"Your answers: {completed}")
    print(f"Answer key: {answer_key}\n")

    score_exam(str(completed), str(answer_key))
