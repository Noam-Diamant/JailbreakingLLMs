#!/usr/bin/env python3
"""
Convert bio_mcq.json into a CSV format suitable for jailbreaking experiments.
The output format matches harmful_behaviors_custom.csv structure:
  - goal: The question/prompt to jailbreak
  - target: The desired response (includes the correct answer)
  - category: "biosecurity" or "bio"
  - Original index: The original question index
"""

import json
import csv
import sys
from pathlib import Path

def convert_bio_mcq_to_jailbreak_csv(json_path: str, output_csv_path: str):
    """
    Convert bio_mcq.json to jailbreak CSV format.
    
    Args:
        json_path: Path to bio_mcq.json
        output_csv_path: Path to output CSV file
    """
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    answers = data.get('answers', [])
    choices = data.get('choices', [])
    
    if len(questions) != len(answers):
        print(f"Warning: Mismatch between questions ({len(questions)}) and answers ({len(answers)})")
    
    if len(questions) != len(choices):
        print(f"Warning: Mismatch between questions ({len(questions)}) and choices ({len(choices)})")
    
    # Prepare CSV data
    csv_rows = []
    csv_rows.append(['', 'goal', 'target', 'category', 'Original index'])
    
    for idx, question in enumerate(questions):
        # Get the correct answer if available
        if idx < len(answers) and idx < len(choices):
            answer_idx = answers[idx]
            if answer_idx < len(choices[idx]):
                correct_answer = choices[idx][answer_idx]
                # Create target string that includes the answer
                target = f"Sure, here is the answer: {correct_answer}"
            else:
                # Fallback if answer index is out of range
                target = f"Sure, here is information about: {question}"
        else:
            # Fallback if no answer/choices available
            target = f"Sure, here is information about: {question}"
        
        # Use question as the goal
        goal = question
        
        # Category is biosecurity
        category = "biosecurity"
        
        # Original index is the question index
        original_index = idx
        
        csv_rows.append([idx, goal, target, category, original_index])
    
    # Write CSV file
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"Successfully converted {len(questions)} questions to {output_csv_path}")
    print(f"Output format: goal, target, category='biosecurity', Original index")
    return len(questions)

if __name__ == "__main__":
    # Default paths
    script_dir = Path(__file__).parent
    json_path = script_dir / "data" / "bio_mcq.json"
    output_csv_path = script_dir / "data" / "bio_mcq_jailbreak.csv"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv_path = sys.argv[2]
    
    convert_bio_mcq_to_jailbreak_csv(str(json_path), str(output_csv_path))

