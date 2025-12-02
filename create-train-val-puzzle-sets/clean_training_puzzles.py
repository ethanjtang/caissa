
'''
Use to remove the puzzle tags '<puzzle-start/>' and '<puzzle-end/>' from an existing data file.

You can keep the tags, but I wanted to simplify the model as much as possible. 
Caissa v.0 uses only position (FEN string) + best move pairs as its input data for training.
'''
puzzle_path = r"path_to_training_puzzles"
line_count = 0
with open(puzzle_path, 'r') as input_file:
    with open('data.txt', 'w') as output_file:
        for line in input_file:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"Processed {line_count} lines...")
            # Remove lines that contain puzzle-start or puzzle-end tags
            if '<puzzle-start/>' not in line and '<puzzle-end/>' not in line:
                output_file.write(line)

print(f"Finished! Processed {line_count} total lines")
print("Cleaned puzzles saved to data.txt")
