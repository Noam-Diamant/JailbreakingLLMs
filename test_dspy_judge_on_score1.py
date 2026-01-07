#!/usr/bin/env python3
"""
Script to test DSPy judge on rows from bio_judge_scoring.csv where judge_scores == 1.
Calculates success rate (how many the judge correctly identifies as 1).
"""
import pandas as pd
import argparse
import sys
import os
import dspy

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from compile_dspy_judge import load_judge, setup_dspy_lm
from config import Model, HF_MODEL_NAMES


def test_dspy_judge(csv_path, judge_path, num_rows, judge_model_name, api_base=None):
    """
    Test DSPy judge on rows with score 1 from CSV.
    
    Args:
        csv_path: Path to bio_judge_scoring.csv
        judge_path: Path to compiled DSPy judge JSON
        num_rows: Number of rows to analyze
        judge_model_name: Model name for DSPy judge
        api_base: Optional API base URL for judge model
    """
    print("=" * 80)
    print("Testing DSPy Judge on Score=1 Examples")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Filter for rows where judge_scores == 1
    score1_rows = df[df['judge_scores'] == 1].copy()
    print(f"Rows with judge_scores == 1: {len(score1_rows)}")
    
    if len(score1_rows) == 0:
        print("ERROR: No rows with judge_scores == 1 found!")
        return
    
    # Limit to num_rows
    if num_rows > 0:
        score1_rows = score1_rows.head(num_rows)
        print(f"Analyzing first {num_rows} rows with score=1...")
    else:
        print(f"Analyzing all {len(score1_rows)} rows with score=1...")
    
    # Setup DSPy language model
    print(f"\nSetting up DSPy language model: {judge_model_name}")
    if api_base:
        print(f"Using HTTP API base: {api_base}")
    else:
        print("No API base provided - will try to auto-detect or use local")
    
    try:
        setup_dspy_lm(
            model_name=judge_model_name,
            use_vllm=True,
            api_base=api_base
        )
    except Exception as e:
        print(f"ERROR setting up DSPy LM: {e}")
        print("Make sure the judge model HTTP server is running or provide --api-base")
        return
    
    # Load compiled judge
    print(f"\nLoading compiled judge from {judge_path}...")
    try:
        dspy_judge = load_judge(judge_path)
    except Exception as e:
        print(f"ERROR loading judge: {e}")
        return
    
    # Test on each row
    print(f"\n{'=' * 80}")
    print("Testing judge on examples...")
    print(f"{'=' * 80}\n")
    
    correct = 0
    total = len(score1_rows)
    results = []
    
    for idx, row in score1_rows.iterrows():
        attack_prompt = row['attacker_prompt']
        target_response = row['target_response']
        original_question = row['original_question']
        original_answer = row['original_answer']
        ground_truth = int(row['judge_scores'])
        
        # Run DSPy judge
        try:
            prediction = dspy_judge(
                attack_prompt=attack_prompt,
                target_response=target_response,
                original_question=original_question,
                original_answer=original_answer
            )
            
            # Extract score
            try:
                predicted_score = int(prediction.knowledge_score)
                predicted_score = 1 if predicted_score >= 1 else 0
            except (ValueError, AttributeError, TypeError):
                print(f"Warning: Could not parse score from prediction: {prediction}")
                predicted_score = 0
            
            # Check if correct
            is_correct = (predicted_score == ground_truth)
            if is_correct:
                correct += 1
            
            results.append({
                'row_idx': idx,
                'ground_truth': ground_truth,
                'predicted': predicted_score,
                'correct': is_correct,
                'reasoning': getattr(prediction, 'reasoning', 'N/A')
            })
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Row {idx}: Ground truth={ground_truth}, Predicted={predicted_score}, Correct={is_correct}")
            
        except Exception as e:
            print(f"✗ Row {idx}: ERROR - {e}")
            results.append({
                'row_idx': idx,
                'ground_truth': ground_truth,
                'predicted': -1,  # Error indicator
                'correct': False,
                'reasoning': f"ERROR: {e}"
            })
    
    # Calculate success rate
    success_rate = (correct / total) * 100 if total > 0 else 0
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total examples tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"{'=' * 80}\n")
    
    # Show some examples of incorrect predictions
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\nExamples of incorrect predictions ({len(incorrect)} total):")
        print("-" * 80)
        for i, result in enumerate(incorrect[:5], 1):  # Show first 5
            row = score1_rows.loc[result['row_idx']]
            print(f"\n{i}. Row {result['row_idx']}:")
            print(f"   Ground truth: {result['ground_truth']}")
            print(f"   Predicted: {result['predicted']}")
            print(f"   Attack prompt (first 100 chars): {row['attacker_prompt'][:100]}...")
            print(f"   Target response (first 100 chars): {row['target_response'][:100]}...")
            print(f"   Reasoning: {result['reasoning'][:200] if result['reasoning'] != 'N/A' else 'N/A'}...")
    
    return success_rate, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test DSPy judge on rows with score=1 from bio_judge_scoring.csv"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="data/bio_judge_scoring.csv",
        help="Path to bio_judge_scoring.csv"
    )
    parser.add_argument(
        "--judge-path",
        type=str,
        default="saved_judges/dspy_judge_optimized_light.json",
        help="Path to compiled DSPy judge JSON"
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=50,
        help="Number of rows with score=1 to analyze (0 = all)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="qwen2-57b-a14b-instruct-gptq-int4",
        help="Model name for DSPy judge"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="HTTP API base URL for judge model (e.g., http://localhost:8005/v1)"
    )
    
    args = parser.parse_args()
    
    test_dspy_judge(
        csv_path=args.csv_file,
        judge_path=args.judge_path,
        num_rows=args.num_rows,
        judge_model_name=args.judge_model,
        api_base=args.api_base
    )

