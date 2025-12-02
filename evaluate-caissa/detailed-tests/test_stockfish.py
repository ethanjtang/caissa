import os
import chess
import chess.engine
from typing import List, Dict, Optional
import json
import time

'''
Test the Stockfish 17 engine with thinktime=1.0s on chess puzzles.
Uses the same puzzles as tested in openai_models_results.json for a fair comparison.
'''

class StockfishChessTester:
    def __init__(self, engine_path: str, think_time: float = 1.0):
        """Initialize the Stockfish chess tester.

        Args:
            engine_path: Path to Stockfish executable
            think_time: Time limit for engine thinking in seconds
        """
        self.engine_path = engine_path
        self.think_time = think_time
        self.engine = None

    def start_engine(self):
        """Start the Stockfish engine."""
        print(f"Starting Stockfish engine from {self.engine_path}...")
        print(f"Think time: {self.think_time}s per move")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        print("Engine started successfully!")

    def stop_engine(self):
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            print("Engine stopped.")

    def load_puzzles_from_openai_results(self, openai_results_file: str) -> Dict[str, List[List[Dict]]]:
        """Load the exact same puzzles that were tested in OpenAI results.

        Args:
            openai_results_file: Path to openai_models_results.json

        Returns:
            Dictionary mapping puzzle_type to list of puzzles
        """
        with open(openai_results_file, 'r', encoding='utf-8') as f:
            openai_results = json.load(f)

        puzzles_by_type = {}

        for puzzle_type, models_data in openai_results.items():
            # Get the first model's details to extract puzzles
            first_model_key = list(models_data.keys())[0]
            details = models_data[first_model_key]['details']

            # Convert details back to puzzle format
            puzzles = []
            for puzzle_detail in details:
                puzzle_moves = []
                for move_detail in puzzle_detail['moves']:
                    puzzle_moves.append({
                        'fen': move_detail['fen'],
                        'best_move': move_detail['correct_move']
                    })
                if puzzle_moves:
                    puzzles.append(puzzle_moves)

            puzzles_by_type[puzzle_type] = puzzles

        return puzzles_by_type

    def query_engine(self, fen: str) -> tuple:
        """Query Stockfish engine for best move.

        Returns:
            tuple: (best_move_uci, response_time)
        """
        board = chess.Board(fen)

        start_time = time.time()
        result = self.engine.play(board, chess.engine.Limit(time=self.think_time))
        response_time = time.time() - start_time

        if result.move:
            return (result.move.uci(), response_time)
        else:
            return (None, response_time)

    def test_engine_on_puzzles(self, puzzles: List[List[Dict]]) -> Dict:
        """Test Stockfish engine on puzzles.

        Args:
            puzzles: List of puzzles, where each puzzle is a list of positions

        Returns:
            Dictionary with move-level and puzzle-level accuracy
        """
        # Count total moves and puzzles
        total_moves = sum(len(puzzle) for puzzle in puzzles)
        total_puzzles = len(puzzles)

        results = {
            'moves_correct': 0,
            'moves_valid': 0,
            'moves_total': total_moves,
            'puzzles_correct': 0,
            'puzzles_total': total_puzzles,
            'total_response_time': 0.0,
            'details': []
        }

        print(f"\nTesting Stockfish 17 (thinktime={self.think_time}s) on {total_puzzles} puzzles ({total_moves} total moves)...")

        for puzzle_idx, puzzle in enumerate(puzzles):
            if (puzzle_idx + 1) % 10 == 0:
                print(f"  Progress: {puzzle_idx + 1}/{total_puzzles} puzzles")

            puzzle_all_correct = True
            puzzle_details = []

            for move_idx, position in enumerate(puzzle):
                fen = position['fen']
                correct_move = position['best_move']

                # Query engine
                engine_move, response_time = self.query_engine(fen)

                # Accumulate response time
                results['total_response_time'] += response_time

                # Check validity
                board = chess.Board(fen)
                is_valid = False
                is_correct = False

                if engine_move:
                    try:
                        move = chess.Move.from_uci(engine_move)
                        is_valid = move in board.legal_moves
                    except:
                        is_valid = False

                    is_correct = (engine_move == correct_move)

                if is_valid:
                    results['moves_valid'] += 1
                if is_correct:
                    results['moves_correct'] += 1
                else:
                    puzzle_all_correct = False

                # Print result
                status = "✓" if is_correct else ("?" if is_valid else "✗")
                print(f"  {status} Puzzle {puzzle_idx+1} Move {move_idx+1}/{len(puzzle)}: Correct={correct_move}, Engine={engine_move} [Time: {response_time:.2f}s]")
                puzzle_details.append({
                    'fen': fen,
                    'correct_move': correct_move,
                    'engine_move': engine_move,
                    'is_valid': is_valid,
                    'is_correct': is_correct,
                    'response_time': response_time
                })

            # Record puzzle-level correctness
            if puzzle_all_correct:
                results['puzzles_correct'] += 1
                print(f"  ✓✓ Puzzle {puzzle_idx+1} FULLY SOLVED")
            else:
                print(f"  ✗✗ Puzzle {puzzle_idx+1} incomplete")

            results['details'].append({
                'puzzle_index': puzzle_idx,
                'moves': puzzle_details,
                'puzzle_solved': puzzle_all_correct
            })

        return results

    def print_results(self, results: Dict, puzzle_type: str):
        """Print results summary."""
        moves_total = results['moves_total']
        moves_correct = results['moves_correct']
        moves_valid = results['moves_valid']
        puzzles_total = results['puzzles_total']
        puzzles_correct = results['puzzles_correct']
        total_response_time = results.get('total_response_time', 0.0)

        print(f"\n{'='*80}")
        print(f"Results for Stockfish 17 (thinktime={self.think_time}s) on {puzzle_type}")
        print(f"{'='*80}")
        print(f"MOVE-BY-MOVE ACCURACY:")
        print(f"  Total moves: {moves_total}")
        print(f"  Correct moves: {moves_correct} ({100*moves_correct/moves_total:.2f}%)")
        print(f"  Valid moves: {moves_valid} ({100*moves_valid/moves_total:.2f}%)")
        print(f"\nPUZZLE-LEVEL ACCURACY:")
        print(f"  Total puzzles: {puzzles_total}")
        print(f"  Fully solved puzzles: {puzzles_correct} ({100*puzzles_correct/puzzles_total:.2f}%)")
        print(f"\nRESPONSE TIME:")
        print(f"  Total response time: {total_response_time:.2f}s")
        if moves_total > 0:
            avg_time = total_response_time / moves_total
            print(f"  Average time per move: {avg_time:.2f}s")
        if puzzles_total > 0:
            avg_time_puzzle = total_response_time / puzzles_total
            print(f"  Average time per puzzle: {avg_time_puzzle:.2f}s")
        print(f"{'='*80}\n")


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Path to Stockfish engine
    engine_path = os.path.join(project_root, "the-big-fish", "stockfish-windows-x86-64-avx2.exe")

    if not os.path.exists(engine_path):
        print(f"Error: Stockfish engine not found at {engine_path}")
        return

    # Path to OpenAI results file
    openai_results_file = os.path.join(project_root, "openai_models_n=100_results.json")

    if not os.path.exists(openai_results_file):
        print(f"Error: OpenAI results file not found at {openai_results_file}")
        print("Please run test_openai_models_reasoning.py first to generate the comparison baseline.")
        return

    # Initialize tester with 1 second think time
    tester = StockfishChessTester(engine_path, think_time=1.0)
    tester.start_engine()

    # Store all results
    all_results = {}

    try:
        # Load the exact same puzzles tested by OpenAI models
        print(f"\nLoading puzzles from {openai_results_file}...")
        puzzles_by_type = tester.load_puzzles_from_openai_results(openai_results_file)
        print(f"Loaded {len(puzzles_by_type)} puzzle types")

        # Test each puzzle type
        for puzzle_type, puzzles in puzzles_by_type.items():
            print(f"\n{'='*80}")
            print(f"Testing {puzzle_type} puzzles")
            print(f"{'='*80}")

            total_moves = sum(len(p) for p in puzzles)
            print(f"Testing {len(puzzles)} puzzles ({total_moves} total moves)")

            # Test engine on puzzles
            results = tester.test_engine_on_puzzles(puzzles)
            tester.print_results(results, puzzle_type)

            # Store results
            all_results[puzzle_type] = {
                'engine': 'Stockfish 17',
                'think_time': tester.think_time,
                'summary': {
                    'moves_total': results['moves_total'],
                    'moves_correct': results['moves_correct'],
                    'moves_valid': results['moves_valid'],
                    'move_accuracy': 100 * results['moves_correct'] / results['moves_total'] if results['moves_total'] > 0 else 0,
                    'move_valid_rate': 100 * results['moves_valid'] / results['moves_total'] if results['moves_total'] > 0 else 0,
                    'puzzles_total': results['puzzles_total'],
                    'puzzles_correct': results['puzzles_correct'],
                    'puzzle_accuracy': 100 * results['puzzles_correct'] / results['puzzles_total'] if results['puzzles_total'] > 0 else 0,
                    'total_response_time': results['total_response_time'],
                    'avg_response_time_per_move': results['total_response_time'] / results['moves_total'] if results['moves_total'] > 0 else 0,
                    'avg_response_time_per_puzzle': results['total_response_time'] / results['puzzles_total'] if results['puzzles_total'] > 0 else 0
                },
                'details': results['details']
            }

    finally:
        # Always stop the engine
        tester.stop_engine()

    # Save all results to project root directory
    output_file = os.path.join(project_root, "stockfish17_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    # Print summary
    print(f"\n{'='*120}")
    print("SUMMARY - Stockfish 17 on All Puzzle Types")
    print(f"{'='*120}")
    print(f"{'Puzzle Type':<20} {'Move Acc':<20} {'Puzzle Acc':<15} {'Avg Time/Move':<15} {'Total Puzzles':<15}")
    print(f"{'-'*120}")

    for puzzle_type in puzzles_by_type.keys():
        if puzzle_type not in all_results:
            continue

        summary = all_results[puzzle_type]['summary']
        move_acc = f"{summary['move_accuracy']:>6.2f}% ({summary['moves_correct']}/{summary['moves_total']})"
        puzzle_acc = f"{summary['puzzle_accuracy']:>6.2f}% ({summary['puzzles_correct']}/{summary['puzzles_total']})"
        avg_time = f"{summary['avg_response_time_per_move']:.2f}s"
        total_puzzles = f"{summary['puzzles_total']}"

        print(f"{puzzle_type:<20} {move_acc:<20} {puzzle_acc:<15} {avg_time:<15} {total_puzzles}")

    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
