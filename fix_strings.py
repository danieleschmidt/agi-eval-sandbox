#!/usr/bin/env python3
"""Fix escaped string literals in Python files."""

import re
import sys
from pathlib import Path

def fix_file(file_path):
    """Fix escaped strings in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix escaped quotes in strings
        # Pattern matches \"...\" and replaces with "..."
        content = re.sub(r'\\"([^"]*)\\"', r'"\1"', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all Python files."""
    files_to_fix = [
        "src/agi_eval_sandbox/core/benchmarks.py",
        "src/agi_eval_sandbox/core/evaluator.py", 
        "src/agi_eval_sandbox/core/logging_config.py",
        "src/agi_eval_sandbox/core/validation.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            if fix_file(path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()