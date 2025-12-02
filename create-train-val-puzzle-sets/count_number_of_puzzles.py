import sys
from pathlib import Path

'''
Use to count number of puzzles in a file or directory. 
Used to generate some hand-picked fresh-off-the-vine statistics for my project presentation and paper.
'''

# Pretty sure Linux (and maybe Windows) has multiple built-in functions for this but I am a moron so here we go...

def count_occurrences(file_path, line_of_interest):
    """Count the number of occurrences of a string in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            count = content.count(line_of_interest)
            return count
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None


def count_occurrences_in_directory(directory_path, line_of_interest, file_pattern='*'):
    """Count occurrences of a string across all files in a directory."""
    total_count = 0
    files_processed = 0

    try:
        directory = Path(directory_path)

        if not directory.exists():
            print(f"Error: Directory '{directory_path}' not found.")
            return None

        if not directory.is_dir():
            print(f"Error: '{directory_path}' is not a directory.")
            return None

        # Find all files matching the pattern
        files = list(directory.glob(file_pattern))

        if not files:
            print(f"No files found matching pattern '{file_pattern}' in '{directory_path}'")
            return 0

        print(f"Processing {len(files)} file(s)...\n")

        for file_path in files:
            if file_path.is_file():
                count = count_occurrences(str(file_path), line_of_interest)
                if count is not None:
                    files_processed += 1
                    if count > 0:
                        print(f"{file_path.name}: {count}")
                    total_count += count

        print(f"\nProcessed {files_processed} file(s)")
        return total_count

    except Exception as e:
        print(f"Error processing directory: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Count in a single file:")
        print("    python count_number_of_puzzles.py <file_path> [word]")
        print("  Count in a directory:")
        print("    python count_number_of_puzzles.py --dir <directory_path> [word] [file_pattern]")
        print("\nExamples:")
        print("  python count_number_of_puzzles.py puzzles.txt")
        print("  python count_number_of_puzzles.py puzzles.txt '<puzzle-start/>'")
        print("  python count_number_of_puzzles.py --dir ./puzzles FEN")
        print("  python count_number_of_puzzles.py --dir ./puzzles FEN '*.txt'")
        sys.exit(1)

    # Default word to search for
    line_of_interest = 'FEN'

    # Check if directory mode
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print("Error: Directory path required when using --dir")
            sys.exit(1)

        directory_path = sys.argv[2]

        # Optional: custom word to search for
        if len(sys.argv) >= 4:
            line_of_interest = sys.argv[3]

        # Optional: file pattern
        file_pattern = '*'
        if len(sys.argv) >= 5:
            file_pattern = sys.argv[4]

        print(f"Counting occurrences of '{line_of_interest}' in directory '{directory_path}'")
        print(f"File pattern: {file_pattern}\n")

        count = count_occurrences_in_directory(directory_path, line_of_interest, file_pattern)

        if count is not None:
            print(f"\nTotal occurrences: {count}")
    else:
        # Single file mode
        file_path = sys.argv[1]

        # Optional: custom word to search for
        if len(sys.argv) >= 3:
            line_of_interest = sys.argv[2]

        print(f"Counting occurrences of '{line_of_interest}' in file '{file_path}'\n")

        count = count_occurrences(file_path, line_of_interest)

        if count is not None:
            print(f"Number of occurrences: {count}")