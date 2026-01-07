"""
Script to process bio_judge_scoring.csv and convert it to DSPy Examples format.
This prepares training data for DSPy judge optimization with balanced classes.
"""
import pandas as pd
import dspy
import json
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def load_csv_to_dspy_examples(csv_path: str) -> List[dspy.Example]:
    """
    Load CSV file and convert to DSPy Examples.
    
    Args:
        csv_path: Path to bio_judge_scoring.csv
        
    Returns:
        List of dspy.Example objects ready for DSPy optimization
    """
    df = pd.read_csv(csv_path)
    
    examples = []
    for _, row in df.iterrows():
        # Extract fields
        attacker_prompt = str(row['attacker_prompt'])
        target_response = str(row['target_response'])
        judge_score = int(row['judge_scores'])  # 0 or 1
        original_question = str(row['original_question'])
        original_answer = str(row['original_answer'])
        
        # Create DSPy Example
        # The signature will be: attack_prompt, target_response, original_question, original_answer -> knowledge_score
        example = dspy.Example(
            attack_prompt=attacker_prompt,
            target_response=target_response,
            original_question=original_question,
            original_answer=original_answer,
            knowledge_score=judge_score  # This is the label (0 or 1)
        ).with_inputs('attack_prompt', 'target_response', 'original_question', 'original_answer')
        
        examples.append(example)
    
    return examples

def create_balanced_dataset(examples: List[dspy.Example]) -> List[dspy.Example]:
    """
    Create a balanced dataset with 50% score 1 and 50% score 0.
    
    Args:
        examples: List of all examples
        
    Returns:
        Balanced list of examples (50% score 1, 50% score 0)
    """
    # Separate examples by score
    score_0_examples = [ex for ex in examples if ex.knowledge_score == 0]
    score_1_examples = [ex for ex in examples if ex.knowledge_score == 1]
    
    print(f"Original dataset:")
    print(f"  Score 0: {len(score_0_examples)}")
    print(f"  Score 1: {len(score_1_examples)}")
    
    # Find the minimum count to balance
    min_count = min(len(score_0_examples), len(score_1_examples))
    
    # Sample equal numbers from each class
    import random
    random.Random(42).shuffle(score_0_examples)  # Fixed seed for reproducibility
    random.Random(42).shuffle(score_1_examples)  # Fixed seed for reproducibility
    
    balanced_score_0 = score_0_examples[:min_count]
    balanced_score_1 = score_1_examples[:min_count]
    
    # Combine and shuffle
    balanced_examples = balanced_score_0 + balanced_score_1
    random.Random(42).shuffle(balanced_examples)
    
    print(f"\nBalanced dataset:")
    print(f"  Score 0: {len(balanced_score_0)} ({len(balanced_score_0)/len(balanced_examples)*100:.1f}%)")
    print(f"  Score 1: {len(balanced_score_1)} ({len(balanced_score_1)/len(balanced_examples)*100:.1f}%)")
    print(f"  Total: {len(balanced_examples)}")
    
    return balanced_examples

def stratified_split(examples: List[dspy.Example], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Split examples into train/val/test sets with stratified splitting (maintains class balance).
    
    Args:
        examples: List of examples to split
        train_ratio: Fraction for training (default: 0.6)
        val_ratio: Fraction for validation (default: 0.2)
        test_ratio: Fraction for test (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    # Convert to DataFrame for easier handling
    data = []
    for ex in examples:
        data.append({
            'attack_prompt': ex.attack_prompt,
            'target_response': ex.target_response,
            'original_question': ex.original_question,
            'original_answer': ex.original_answer,
            'knowledge_score': ex.knowledge_score
        })
    df = pd.DataFrame(data)
    
    # Extract labels for stratification
    y = df['knowledge_score']
    
    # First split: train vs (val+test)
    # Calculate sizes
    train_size = train_ratio
    val_test_size = val_ratio + test_ratio
    val_size_in_val_test = val_ratio / val_test_size  # Proportion of val in (val+test)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y,
        test_size=val_test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size_in_val_test),  # test_size is the proportion of test in temp
        stratify=y_temp,
        random_state=random_state
    )
    
    # Convert back to DSPy Examples
    train_examples = []
    for _, row in X_train.iterrows():
        ex = dspy.Example(
            attack_prompt=row['attack_prompt'],
            target_response=row['target_response'],
            original_question=row['original_question'],
            original_answer=row['original_answer'],
            knowledge_score=row['knowledge_score']
        ).with_inputs('attack_prompt', 'target_response', 'original_question', 'original_answer')
        train_examples.append(ex)
    
    val_examples = []
    for _, row in X_val.iterrows():
        ex = dspy.Example(
            attack_prompt=row['attack_prompt'],
            target_response=row['target_response'],
            original_question=row['original_question'],
            original_answer=row['original_answer'],
            knowledge_score=row['knowledge_score']
        ).with_inputs('attack_prompt', 'target_response', 'original_question', 'original_answer')
        val_examples.append(ex)
    
    test_examples = []
    for _, row in X_test.iterrows():
        ex = dspy.Example(
            attack_prompt=row['attack_prompt'],
            target_response=row['target_response'],
            original_question=row['original_question'],
            original_answer=row['original_answer'],
            knowledge_score=row['knowledge_score']
        ).with_inputs('attack_prompt', 'target_response', 'original_question', 'original_answer')
        test_examples.append(ex)
    
    # Print statistics
    def print_split_stats(name, examples_list):
        scores = [ex.knowledge_score for ex in examples_list]
        score_0_count = scores.count(0)
        score_1_count = scores.count(1)
        total = len(scores)
        print(f"\n{name} split:")
        print(f"  Total: {total}")
        print(f"  Score 0: {score_0_count} ({score_0_count/total*100:.1f}%)")
        print(f"  Score 1: {score_1_count} ({score_1_count/total*100:.1f}%)")
    
    print_split_stats("Train", train_examples)
    print_split_stats("Validation", val_examples)
    print_split_stats("Test", test_examples)
    
    return train_examples, val_examples, test_examples

def save_examples_to_json(examples: List[dspy.Example], output_path: str):
    """Save DSPy examples to JSON file for later loading."""
    data = []
    for ex in examples:
        data.append({
            'attack_prompt': ex.attack_prompt,
            'target_response': ex.target_response,
            'original_question': ex.original_question,
            'original_answer': ex.original_answer,
            'knowledge_score': ex.knowledge_score
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(examples)} examples to {output_path}")

def load_examples_from_json(json_path: str) -> List[dspy.Example]:
    """Load DSPy examples from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        example = dspy.Example(
            attack_prompt=item['attack_prompt'],
            target_response=item['target_response'],
            original_question=item['original_question'],
            original_answer=item['original_answer'],
            knowledge_score=item['knowledge_score']
        ).with_inputs('attack_prompt', 'target_response', 'original_question', 'original_answer')
        examples.append(example)
    
    return examples

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare DSPy training data from CSV with balanced classes")
    parser.add_argument("--csv-file", type=str, default="data/bio_judge_scoring.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output-json", type=str, default="data/dspy_judge_examples.json",
                       help="Path to output JSON file")
    parser.add_argument("--train-ratio", type=float, default=0.6,
                       help="Fraction of data to use for training (default: 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Fraction of data to use for test (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load and convert CSV to DSPy examples
    print(f"Loading CSV from {args.csv_file}...")
    examples = load_csv_to_dspy_examples(args.csv_file)
    print(f"Loaded {len(examples)} examples")
    
    # Create balanced dataset (50% score 0, 50% score 1)
    balanced_examples = create_balanced_dataset(examples)
    
    # Stratified split into train/val/test (60-20-20)
    train_examples, val_examples, test_examples = stratified_split(
        balanced_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # Save all splits
    train_path = args.output_json.replace('.json', '_train.json')
    val_path = args.output_json.replace('.json', '_val.json')
    test_path = args.output_json.replace('.json', '_test.json')
    
    save_examples_to_json(train_examples, train_path)
    save_examples_to_json(val_examples, val_path)
    save_examples_to_json(test_examples, test_path)
    
    # Also save balanced combined for convenience
    save_examples_to_json(balanced_examples, args.output_json)
    
    print(f"\n{'='*80}")
    print("Data preparation complete!")
    print(f"{'='*80}")
    print(f"Train data: {train_path} ({len(train_examples)} examples)")
    print(f"Validation data: {val_path} ({len(val_examples)} examples)")
    print(f"Test data: {test_path} ({len(test_examples)} examples)")
    print(f"All data: {args.output_json} ({len(balanced_examples)} examples)")


