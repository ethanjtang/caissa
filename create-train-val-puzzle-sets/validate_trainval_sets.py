from pathlib import Path

'''
Use to validate that training and validation sets of puzzles do not share any common positions.
'''

data_path = r"\puzzles\data.txt"
validation_path = r"\puzzles\validation_puzzles"

def extract_fens_from_training(file_path):
    """Extract FENs from training data (cleaned format without tags)"""
    fens = {}
    line_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  Processed {line_count} lines...")
            line = line.strip()
            if line.startswith('FEN: '):
                fen = line[5:]  # Remove 'FEN: ' prefix
                fens[fen] = fens.get(fen, 0) + 1
    print(f"  Total lines processed: {line_count}")
    return fens

def extract_fens_from_validation(file_path):
    """Extract FENs from validation data (old format with tags)"""
    fens = {}
    line_count = 0
    with open(file_path, 'r') as f:
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
                fens[fen] = fens.get(fen, 0) + 1
    return fens

def main():
    # Extract FENs from training data
    print("Reading training data from data.txt...")
    training_fens = extract_fens_from_training(data_path)
    print(f"Found {len(training_fens)} unique FENs in training data")

    # Extract FENs from all validation files
    print("\nReading validation data...")
    validation_fens = {}
    validation_dir = Path(validation_path)

    if not validation_dir.exists():
        print(f"Error: {validation_dir} directory not found!")
        return

    validation_files = list(validation_dir.glob('validation_puzzles_*.txt'))
    print(f"Found {len(validation_files)} validation files\n")

    for i, val_file in enumerate(validation_files, 1):
        print(f"  [{i}/{len(validation_files)}] Processing {val_file.name}...")
        file_fens = extract_fens_from_validation(val_file)
        print(f"    Found {len(file_fens)} unique FENs in this file")
        for fen, count in file_fens.items():
            validation_fens[fen] = validation_fens.get(fen, 0) + count

    print(f"\nFound {len(validation_fens)} unique FENs in validation data")

    # Combine all FENs and count occurrences
    print("\nChecking for overlaps...")
    all_fens = {}

    # Add training FENs
    for fen, count in training_fens.items():
        all_fens[fen] = all_fens.get(fen, 0) + count

    # Add validation FENs
    for fen, count in validation_fens.items():
        all_fens[fen] = all_fens.get(fen, 0) + count

    # Find FENs with count > 1 (duplicates across sets)
    duplicates = {fen: count for fen, count in all_fens.items() if count > 1}

    if duplicates:
        print(f"\n\nFound {len(duplicates)} FENs that appear in both training and validation sets!")
        print("\n\nDuplicate FENs:")
        for fen, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            in_training = fen in training_fens
            in_validation = fen in validation_fens
            print(f"  Count: {count} | Training: {in_training} | Validation: {in_validation}")
            print(f"  FEN: {fen}")
            print()
    else:
        print("\n\nNo overlapping FENs found! Training and validation sets are disjoint.")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Training FENs:    {len(training_fens)}")
    print(f"Validation FENs:  {len(validation_fens)}")
    print(f"Overlapping FENs: {len(duplicates)}")
    print(f"Total unique:     {len(all_fens)}")

if __name__ == '__main__':
    main()
