import tiktoken
import os
from collections import Counter

'''
Use to determine context window for training LLM from maximum token count for a position + best move pair.

You can increase the context window to be larger if you want your model to have context from previous positions.
I didn't because I wanted the LLM Caissa to be able to play searchless chess -> to capture "intuition".
'''

# CONTEXT WINDOW = 128 since the maximum puzzles are around 80-90 tokens

def analyze_puzzle_token_lengths(puzzle_file):
    """
    Analyze the token lengths of puzzles to determine max context window needed.
    Uses GPT-2 tokenizer (same as nanoGPT).

    Each puzzle has format:
    Line 1: FEN
    Line 2: Best move (UCI)
    Line 3: Blank line
    """
    # Initialize GPT-2 tokenizer (same as used in nanoGPT)
    enc = tiktoken.get_encoding("gpt2")

    token_lengths = []
    max_tokens = 0
    max_puzzle = None

    print("Analyzing puzzle token lengths...\n")

    with open(puzzle_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process puzzles in groups of 3 lines
    puzzle_count = 0
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break

        fen = lines[i].strip()
        move = lines[i + 1].strip()

        # Create the full puzzle text (FEN + move)
        puzzle_text = f"{fen}\n{move}"

        # Tokenize
        tokens = enc.encode(puzzle_text)
        num_tokens = len(tokens)

        token_lengths.append(num_tokens)
        puzzle_count += 1

        # Print progress every 1000 puzzles
        if puzzle_count % 1000 == 0:
            print(f"Analyzed {puzzle_count:,} puzzles...")

        # Track maximum
        if num_tokens > max_tokens:
            max_tokens = num_tokens
            max_puzzle = (fen, move)

    # Calculate statistics
    avg_tokens = sum(token_lengths) / len(token_lengths)

    # Sort to find percentiles
    sorted_lengths = sorted(token_lengths)
    p50 = sorted_lengths[len(sorted_lengths) // 2]
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
    p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)]

    # Distribution analysis
    token_counter = Counter(token_lengths)

    print(f"Total puzzles analyzed: {len(token_lengths):,}")
    print(f"\nToken Length Statistics:")
    print(f"  Average:  {avg_tokens:.2f} tokens")
    print(f"  Median:   {p50} tokens")
    print(f"  95th percentile: {p95} tokens")
    print(f"  99th percentile: {p99} tokens")
    print(f"  Maximum:  {max_tokens} tokens")

    if max_puzzle:
        print(f"\nMaximum token puzzle:")
        print(f"  FEN:  {max_puzzle[0]}")
        print(f"  Move: {max_puzzle[1]}")
        print(f"  Tokens: {max_tokens}")
    else:
        print("max_puzzle is None, we are fucked")

    print(f"\nRecommended context windows:")
    print(f"  Conservative (99th percentile): {p99} tokens")
    print(f"  Safe (max + buffer):            {max_tokens + 10} tokens")
    print(f"  Optimal (if memory is limited): {p95} tokens (covers 95% of puzzles)")

    # Show distribution
    print(f"\nToken Length Distribution:")
    for length in sorted(token_counter.keys())[:20]:  # Show first 20 unique lengths
        count = token_counter[length]
        percentage = (count / len(token_lengths)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {length:3d} tokens: {bar} {count:,} ({percentage:.2f}%)")

    if len(token_counter) > 20:
        print(f"  ... ({len(token_counter) - 20} more unique lengths)")

    return {
        'avg': avg_tokens,
        'median': p50,
        'p95': p95,
        'p99': p99,
        'max': max_tokens,
        'total_puzzles': len(token_lengths)
    }

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    puzzle_file = os.path.join(script_dir, "training_puzzles.txt")

    try:
        stats = analyze_puzzle_token_lengths(puzzle_file)
    except FileNotFoundError:
        print(f"Error: {puzzle_file} not found.")
        print("Please run extract_puzzles.py first to generate the puzzle file.")