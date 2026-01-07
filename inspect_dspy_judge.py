#!/usr/bin/env python3
"""
Script to inspect the DSPy judge's optimized system prompt and configuration.
"""
import json
import sys
import os

def inspect_dspy_judge(judge_path: str):
    """Load and display the DSPy judge's system prompt and configuration."""
    
    if not os.path.exists(judge_path):
        print(f"Error: Judge file not found at {judge_path}")
        return
    
    with open(judge_path, 'r') as f:
        judge_data = json.load(f)
    
    print("=" * 80)
    print("DSPy Judge System Prompt")
    print("=" * 80)
    
    # Extract the instructions (system prompt)
    if "classify.predict" in judge_data:
        signature = judge_data["classify.predict"].get("signature", {})
        instructions = signature.get("instructions", "No instructions found")
        
        print("\n📝 Optimized System Prompt (Instructions):")
        print("-" * 80)
        print(instructions)
        print("-" * 80)
        
        # Show field descriptions
        print("\n📋 Field Descriptions:")
        print("-" * 80)
        fields = signature.get("fields", [])
        for field in fields:
            prefix = field.get("prefix", "")
            desc = field.get("description", "")
            print(f"  {prefix:20s} {desc}")
        
        # Show few-shot examples
        demos = judge_data["classify.predict"].get("demos", [])
        print(f"\n📚 Few-Shot Examples: {len(demos)} examples")
        print("-" * 80)
        for i, demo in enumerate(demos[:3], 1):  # Show first 3 examples
            print(f"\nExample {i}:")
            print(f"  Attack Prompt: {demo.get('attack_prompt', 'N/A')[:100]}...")
            print(f"  Knowledge Score: {demo.get('knowledge_score', 'N/A')}")
            if 'reasoning' in demo:
                print(f"  Reasoning: {demo.get('reasoning', 'N/A')[:100]}...")
        
        if len(demos) > 3:
            print(f"\n  ... and {len(demos) - 3} more examples")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    judge_path = sys.argv[1] if len(sys.argv) > 1 else "saved_judges/dspy_judge_optimized.json"
    inspect_dspy_judge(judge_path)

