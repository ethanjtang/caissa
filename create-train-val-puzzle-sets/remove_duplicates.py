from pathlib import Path

'''
Use to remove duplicate instances of a position from training set to ensure LLM is not memorizing answers.
'''

data_path = r"uncleaned_data.txt"
validation_path = r"validation_puzzles"
output_path = r"\puzzles\data.txt"

def extract_fens_from_validation_files():
    """Extract all FENs from validation files"""
    validation_fens = set()
    validation_dir = Path(validation_path)

    if not validation_dir.exists():
        print(f"Error: {validation_dir} directory not found!")
        return validation_fens

    validation_files = list(validation_dir.glob('validation_puzzles_*.txt'))
    print(f"Found {len(validation_files)} validation files")

    for i, val_file in enumerate(validation_files, 1):
        print(f"  [{i}/{len(validation_files)}] Reading {val_file.name}...")
        line_count = 0
        with open(val_file, 'r') as f:
            for line in f:
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"    Processed {line_count} lines...")
                line = line.strip()
                # Skip puzzle tags
                if '<puzzle-start/>' in line or '<puzzle-end/>' in line:
                    continue
                if line.startswith('FEN: '):
                    fen = line[5:]  # Remove 'FEN: ' prefix
                    validation_fens.add(fen)

    print(f"\nFound {len(validation_fens)} unique FENs in validation set")
    return validation_fens

def remove_duplicate_fens(training_file, output_file, validation_fens):
    """Remove FENs from training data that exist in validation set"""
    print("\nProcessing training data...")

    removed_count = 0
    kept_count = 0
    line_count = 0
    current_puzzle_lines = []
    current_puzzle_fens = []
    skip_current_puzzle = False

    with open(training_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  Processed {line_count} lines...")

            stripped = line.strip()

            # Check if this is a FEN line
            if stripped.startswith('FEN: '):
                fen = stripped[5:]
                current_puzzle_fens.append(fen)
                current_puzzle_lines.append(line)

                # Check if this FEN is in validation set
                if fen in validation_fens:
                    skip_current_puzzle = True

            # Check if we've reached end of puzzle (empty line or new FEN after moves)
            elif stripped == '' or stripped.startswith('Best move:'):
                current_puzzle_lines.append(line)

                # If it's an empty line and we have accumulated puzzle data
                if stripped == '' and current_puzzle_lines:
                    if skip_current_puzzle:
                        # Skip this puzzle
                        removed_count += len(current_puzzle_fens)
                    else:
                        # Write this puzzle
                        for puzzle_line in current_puzzle_lines:
                            output_f.write(puzzle_line)
                        kept_count += len(current_puzzle_fens)

                    # Reset for next puzzle
                    current_puzzle_lines = []
                    current_puzzle_fens = []
                    skip_current_puzzle = False
            else:
                current_puzzle_lines.append(line)

        # Handle last puzzle if file doesn't end with empty line
        if current_puzzle_lines:
            if not skip_current_puzzle:
                for puzzle_line in current_puzzle_lines:
                    output_f.write(puzzle_line)
                kept_count += len(current_puzzle_fens)
            else:
                removed_count += len(current_puzzle_fens)

    print(f"  Total lines processed: {line_count}")
    print(f"\nRemoved {removed_count} duplicate FEN positions")
    print(f"Kept {kept_count} unique FEN positions")
    print(f"Cleaned data saved to: {output_file}")

def main():
    print("="*70)
    print("REMOVING DUPLICATE FENs FROM TRAINING DATA")
    print("="*70)

    # Step 1: Extract all FENs from validation set
    print("\nStep 1: Reading validation set...")
    validation_fens = extract_fens_from_validation_files()

    # Step 2: Remove duplicates from training data
    print("\nStep 2: Removing duplicates from training data...")
    remove_duplicate_fens(data_path, output_path, validation_fens)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
