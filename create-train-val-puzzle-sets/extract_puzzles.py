import csv
import chess
import random
from collections import defaultdict
import os

'''
The meat and potatoes of this directory .\create-train-val-puzzle-sets.

Reads puzzles from a CSV file and converts them into .txt files, separating puzzles into train and validation sets.
I used Lichess's open-source puzzle database containing a set of 5m+ puzzles:
https://database.lichess.org/#puzzles
'''

def extract_puzzle_positions(csv_file, output_file, validation_dir=None, validation_sample_size=100):
    """
    Extract FEN and best move pairs from Lichess puzzle database.
    Only captures positions where the initial side to move is playing.
    Format: FEN on line 1, Best move on line 2, blank line on line 3.

    If validation_dir is specified, randomly samples puzzles by theme for validation
    and excludes them from the main output file.
    """
    count = 0
    validation_count = 0
    total_puzzles_by_theme = defaultdict(int)  # Track total puzzles per theme

    # First pass: if validation sampling is requested, collect all puzzles by theme
    validation_puzzles = set()  # Store puzzle IDs to exclude from main file
    puzzles_by_theme = defaultdict(list)

    # Prepare log file for sampling output
    sampling_log = []

    if validation_dir:
        print("First pass: Collecting puzzles by theme for validation sampling...")
        sampling_log.append("First pass: Collecting puzzles by theme for validation sampling...\n")

        with open(csv_file, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)

            for row in reader:
                puzzle_id = row['PuzzleId']
                themes_str = row['Themes']
                themes = themes_str.split()

                # Store puzzle ID with its themes
                for theme in themes:
                    puzzles_by_theme[theme].append(puzzle_id)

        # Sample puzzles for each theme
        os.makedirs(validation_dir, exist_ok=True)

        msg = f"\nFound {len(puzzles_by_theme)} unique themes"
        print(msg)
        sampling_log.append(msg + "\n")

        msg = f"Sampling {validation_sample_size} puzzles per theme...\n"
        print(msg)
        sampling_log.append(msg + "\n")

        for theme, puzzle_ids in puzzles_by_theme.items():
            # Sample puzzle IDs
            sample_ids = set(random.sample(puzzle_ids, min(validation_sample_size, len(puzzle_ids))))
            validation_puzzles.update(sample_ids)

            msg = f"Theme '{theme}': sampled {len(sample_ids)} puzzles (out of {len(puzzle_ids)} total)"
            print(msg)
            sampling_log.append(msg + "\n")

        msg = f"\nTotal unique puzzles selected for validation: {len(validation_puzzles)}"
        print(msg)
        sampling_log.append(msg + "\n")

        print("\nSecond pass: Writing puzzles to files...\n")
        sampling_log.append("\nSecond pass: Writing puzzles to files...\n")

    # Second pass: write puzzles to appropriate files
    validation_files = {}
    validation_theme_counts = defaultdict(int)

    with open(csv_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)

        for row in reader:
            puzzle_id = row['PuzzleId']
            fen = row['FEN']
            moves_str = row['Moves']
            themes_str = row['Themes']
            themes = themes_str.split()

            # Track total puzzles processed per theme
            for theme in themes:
                total_puzzles_by_theme[theme] += 1

            # Split moves into a list
            moves = moves_str.split()

            # Create a chess board from the FEN
            board = chess.Board(fen)

            # Check if this puzzle is in validation set
            is_validation = validation_dir and puzzle_id in validation_puzzles

            # The first move (index 0) is the opponent's setup move
            # We want to extract the player's responses: index 1, 3, 5, ...
            # But first we need to play the opponent's move to get the position
            if len(moves) < 2:
                continue  # Skip puzzles with too few moves

            # Play the opponent's first move to set up the position
            board.push(chess.Move.from_uci(moves[0]))

            # Write puzzle start tag
            if is_validation:
                for theme in themes:
                    if theme not in validation_files:
                        assert validation_dir is not None
                        validation_files[theme] = open(
                            os.path.join(validation_dir, f"validation_puzzles_{theme}.txt"),
                            'w',
                            encoding='utf-8'
                        )
                    validation_files[theme].write("<puzzle-start/>\n")
            else:
                f_out.write("<puzzle-start/>\n")

            # Extract every other move starting from index 1 (the player's responses)
            # Index 1, 3, 5, ... are the moves the player needs to find
            for i in range(1, len(moves), 2):
                # Get the best move for this position (in UCI notation)
                best_move_uci = moves[i]

                # Create the FEN for the current position
                current_fen = board.fen()

                # Get all legal moves in UCI notation and sort alphabetically
                legal_moves = sorted([move.uci() for move in board.legal_moves])
                legal_moves_str = " ".join(legal_moves)

                if is_validation:
                    # Write to validation files for each theme
                    for theme in themes:
                        if theme not in validation_files:
                            assert validation_dir is not None
                            validation_files[theme] = open(
                                os.path.join(validation_dir, f"validation_{theme}.txt"),
                                'w',
                                encoding='utf-8'
                            )

                        f = validation_files[theme]
                        f.write(f"FEN: {current_fen}\n")
                        f.write(f"Best move: {best_move_uci}\n")
                        f.write("\n")

                        validation_theme_counts[theme] += 1

                    validation_count += 1
                else:
                    # Write to main output file
                    f_out.write(f"FEN: {current_fen}\n")
                    f_out.write(f"Best move: {best_move_uci}\n")
                    f_out.write("\n")
                    count += 1

                # Print progress every 1000 puzzles
                if (count + validation_count) % 1000 == 0:
                    print(f"Processed {count + validation_count} puzzle positions...")

                # Make the player's move (index i) and opponent's response (index i+1)
                # to advance to the next position
                board.push(chess.Move.from_uci(moves[i]))
                if i + 1 < len(moves):
                    board.push(chess.Move.from_uci(moves[i + 1]))

            # Write puzzle end tag after all positions in this puzzle
            if is_validation:
                for theme in themes:
                    validation_files[theme].write("<puzzle-end/>\n")
            else:
                f_out.write("<puzzle-end/>\n")

    # Close validation files
    for f in validation_files.values():
        f.close()

    print(f"\nExtracted {count} FEN-move pairs to {output_file}")
    if validation_dir:
        print(f"Extracted {validation_count} validation FEN-move pairs to {validation_dir}/")
        print(f"\nValidation puzzles by theme:")
        for theme in sorted(validation_theme_counts.keys()):
            print(f"  {theme}: {validation_theme_counts[theme]} positions")

    # Verify counts by reading the files
    verification_log = []

    separator = "="*80
    print("\n" + separator)
    verification_log.append("\n" + separator + "\n")

    msg = "VERIFICATION: Reading files to verify puzzle and position counts"
    print(msg)
    verification_log.append(msg + "\n")

    print(separator)
    verification_log.append(separator + "\n")

    # Count training file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            training_puzzle_count = content.count('<puzzle-start/>')
            training_position_count = content.count('FEN:')

        msg = f"\nTraining file ({os.path.basename(output_file)}):"
        print(msg)
        verification_log.append(msg + "\n")

        msg = f"  Puzzles: {training_puzzle_count}"
        print(msg)
        verification_log.append(msg + "\n")

        msg = f"  Positions: {training_position_count}"
        print(msg)
        verification_log.append(msg + "\n")

    # Count validation files
    if validation_dir and os.path.exists(validation_dir):
        msg = f"\nValidation files ({os.path.basename(validation_dir)}/):"
        print(msg)
        verification_log.append(msg + "\n")

        validation_file_list = [f for f in os.listdir(validation_dir) if f.startswith('validation_puzzles_')]

        for filename in sorted(validation_file_list):
            filepath = os.path.join(validation_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                puzzle_count = content.count('<puzzle-start/>')
                position_count = content.count('FEN:')

            # Extract theme name from filename
            theme = filename.replace('validation_puzzles_', '').replace('.txt', '')

            # Calculate ratio
            total_for_theme = total_puzzles_by_theme.get(theme, 0)
            ratio = (puzzle_count / total_for_theme * 100) if total_for_theme > 0 else 0

            msg = f"  {theme}:"
            print(msg)
            verification_log.append(msg + "\n")

            msg = f"    Puzzles: {puzzle_count} (out of {total_for_theme} total, {ratio:.2f}%)"
            print(msg)
            verification_log.append(msg + "\n")

            msg = f"    Positions: {position_count}"
            print(msg)
            verification_log.append(msg + "\n")

    print(separator + "\n")
    verification_log.append(separator + "\n")

    # Save logs to files
    if validation_dir:
        # Save sampling log
        sampling_log_file = os.path.join(validation_dir, "sampling_log.txt")
        with open(sampling_log_file, 'w', encoding='utf-8') as f:
            f.writelines(sampling_log)
        print(f"Sampling log saved to {sampling_log_file}")

        # Save verification log
        verification_log_file = os.path.join(validation_dir, "verification_log.txt")
        with open(verification_log_file, 'w', encoding='utf-8') as f:
            f.writelines(verification_log)
        print(f"Verification log saved to {verification_log_file}\n")

    return count

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_file = os.path.join(script_dir, "lichess_db_puzzles.csv")
    output_file = os.path.join(script_dir, "training_puzzles.txt")
    validation_dir = os.path.join(script_dir, "validation_puzzles")  # Set to None to disable validation sampling

    extract_puzzle_positions(csv_file, output_file, validation_dir=validation_dir, validation_sample_size=100)
