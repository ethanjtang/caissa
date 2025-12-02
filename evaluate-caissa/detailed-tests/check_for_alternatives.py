import os
import json
import chess
import chess.engine
from typing import Dict, List, Optional, Any
from copy import deepcopy

'''
Check if alternative moves chosen by LLMs/Stockfish lead to the same mate depth or evaluation.
For mate-in-X puzzles, we check if the model's move achieves mate in the same number of moves.
For non-mate-in-X puzzles, we instead check if moves have equivalent evaluations (within ±0.2 pawns).
If either condition is met, we count it as correct and adjust accuracy scores accordingly.

Updates existing result files in-place with adjusted accuracy.
'''

class AlternativeMoveChecker:
    def __init__(self, stockfish_path: str):
        """Initialize with Stockfish engine for evaluation.

        Args:
            stockfish_path: Path to Stockfish executable
        """
        self.stockfish_path = stockfish_path
        self.engine = None

    def start_engine(self):
        """Start the Stockfish engine."""
        print(f"Starting Stockfish engine from {self.stockfish_path}...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        print("Engine started successfully!")

    def stop_engine(self):
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            print("Engine stopped.")

    def get_move_evaluation(self, fen: str, move_uci: str) -> Optional[float]:
        """Get the centipawn evaluation of a position after making a move.

        Args:
            fen: Position in FEN notation
            move_uci: Move in UCI format

        Returns:
            Evaluation in centipawns (from current player's perspective after the move),
            or None if evaluation cannot be determined.
            For mate scores, returns a large value (e.g., 10000 for mate)
        """
        try:
            if not self.engine:
                return None

            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                return None

            # Make the move
            board.push(move)

            # Check if it's immediate checkmate
            if board.is_checkmate():
                return 10000.0  # Large value for mate

            # Analyze the resulting position
            info = self.engine.analyse(board, chess.engine.Limit(depth=20))

            # Get the score
            score = info.get('score')
            if score:
                relative_score = score.relative
                if relative_score.is_mate():
                    # For mate scores, return a large value
                    mate_value = relative_score.mate()
                    if mate_value is not None:
                        if mate_value < 0:
                            # Opponent is getting mated - good for us
                            return 10000.0 - abs(mate_value)  # Closer mate = higher value
                        else:
                            # We're getting mated - bad
                            return -10000.0 + mate_value
                else:
                    # Regular centipawn score
                    cp_score = relative_score.score()
                    if cp_score is not None:
                        # Negate because it's from opponent's perspective
                        return -cp_score
            return None
        except Exception as e:
            print(f"    Warning: Error evaluating move {move_uci}: {e}")
            return None

    def get_mate_depth(self, fen: str, move_uci: str) -> Optional[int]:
        """Check if a move leads to mate and return mate depth.

        For mate-in-N puzzles:
        - After making the move, the position should show mate in (N-1) from opponent's perspective
        - For mate-in-1: position is immediate checkmate (mate in 0)
        - For mate-in-2: position shows mate in 1 (opponent gets mated in 1)

        Args:
            fen: Position in FEN notation
            move_uci: Move in UCI format

        Returns:
            Mate depth from current player's perspective, or None if not mate
            Returns 1 for immediate checkmate, 2 for mate-in-2, etc.
        """
        try:
            if not self.engine:
                return None

            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                return None

            # Make the move
            board.push(move)

            # Check if it's immediate checkmate
            if board.is_checkmate():
                return 1  # Mate in 1 (immediate checkmate)

            # Analyze the resulting position with sufficient depth for mate detection
            info = self.engine.analyse(board, chess.engine.Limit(depth=20))

            # Check if there's a mate score
            score = info.get('score')
            if score:
                relative_score = score.relative
                if relative_score.is_mate():
                    # Mate score is from the perspective of the side to move (opponent now)
                    # If negative, opponent is getting mated
                    mate_value = relative_score.mate()
                    if mate_value is not None:
                        if mate_value < 0:
                            # Opponent is getting mated in abs(mate_value) moves
                            # So current player mates in abs(mate_value) + 1
                            return abs(mate_value) + 1
            return None
        except Exception as e:
            print(f"    Warning: Error analyzing move {move_uci} in position {fen[:20]}...: {e}")
            return None

    def is_mate_puzzle(self, puzzle_type: str) -> bool:
        """Check if puzzle type is a mate-in-X puzzle."""
        return puzzle_type.startswith('mate_in_')

    def get_expected_mate_depth(self, puzzle_type: str) -> Optional[int]:
        """Extract expected mate depth from puzzle type name.

        Args:
            puzzle_type: e.g., "mate_in_1", "mate_in_2", etc.

        Returns:
            Expected mate depth or None
        """
        if not self.is_mate_puzzle(puzzle_type):
            return None

        try:
            # Extract number from "mate_in_X"
            depth = int(puzzle_type.split('_')[-1])
            return depth
        except:
            return None

    def check_alternative_move(self, fen: str, correct_move: str, model_move: str,
                              expected_mate_depth: Optional[int]) -> tuple[bool, Optional[int], Optional[int], str]:
        """Check if model's alternative move is equivalent to the correct move.

        Checks:
        1. Same mate depth (if either move leads to mate)
        2. Evaluation within ±20 centipawns (for non-mate evaluations)

        Note: Non-mate puzzles can still have mate evaluations if the position has multiple themes.
        We always check mate depth first, regardless of puzzle type.

        Args:
            fen: Position in FEN notation
            correct_move: The labeled "correct" move
            model_move: The move chosen by the model
            expected_mate_depth: Expected mate depth (e.g., 1 for mate-in-1), or None for non-mate puzzles

        Returns:
            Tuple of (is_alternative_correct, correct_move_depth, model_move_depth, reason)
            reason: "exact", "same_mate", "same_eval", or "different"
        """
        # If moves are the same, it's correct (duh!)
        if model_move == correct_move:
            print(f"Identical moves for FEN={fen}, Best move={correct_move}, Model move={model_move}")
            return (True, expected_mate_depth, expected_mate_depth, "exact")

        # Always check mate depth first (non-mate puzzles can have mate evaluations)
        correct_mate_depth = self.get_mate_depth(fen, correct_move)
        model_mate_depth = self.get_mate_depth(fen, model_move)

        # If both moves lead to mate, check if they have the same mate depth
        if correct_mate_depth is not None and model_mate_depth is not None:
            if model_mate_depth == correct_mate_depth:
                print(f"Matching mate depth for FEN={fen}, Best move={correct_move}, Model move={model_move}")
                return (True, correct_mate_depth, model_mate_depth, "same_mate")
            # Different mate depths - this is considered different
            # (e.g., mate in 2 vs mate in 3 should not be equivalent)
            print(f"Non-matching mate depth for FEN={fen}, Best move={correct_move}, Model move={model_move}")
            return (False, correct_mate_depth, model_mate_depth, "different")

        # If one move is mate and the other isn't, they're not equivalent
        if correct_mate_depth is not None or model_mate_depth is not None:
            print(f"Mismatch between mate and non-mate move for FEN={fen}, Best move={correct_move}, Model move={model_move}")
            return (False, correct_mate_depth, model_mate_depth, "different")

        # Neither move is mate, so check centipawn evaluation
        correct_eval = self.get_move_evaluation(fen, correct_move)
        model_eval = self.get_move_evaluation(fen, model_move)

        if correct_eval is not None and model_eval is not None:
            eval_diff = abs(correct_eval - model_eval)
            # Within 20 centipawns
            if eval_diff <= 20.0:
                print(f"Close enough cp={eval_diff} for FEN={fen}, Best move={correct_move}, Model move={model_move}")
                return (True, correct_mate_depth, model_mate_depth, "same_eval")

        return (False, correct_mate_depth, model_mate_depth, "different")

    def process_model_results(self, model_data: Dict[str, Any], puzzle_type: str) -> Dict[str, Any]:
        """Process a model's results and check for alternative solutions.

        For mate puzzles: checks if alternative moves achieve the same mate depth.
        For non-mate puzzles: checks if alternative moves are within ±20 centipawns.

        Args:
            model_data: Model data including summary and details
            puzzle_type: Type of puzzle (e.g., "mate_in_1", "winning_position", etc.)

        Returns:
            Updated model data with adjusted accuracy
        """
        is_mate = self.is_mate_puzzle(puzzle_type)
        expected_mate_depth = self.get_expected_mate_depth(puzzle_type) if is_mate else None

        # Deep copy to avoid modifying original
        updated_data = deepcopy(model_data)

        # Track adjustments
        original_correct = updated_data['summary']['moves_correct']
        original_puzzles_correct = updated_data['summary']['puzzles_correct']

        moves_corrected = 0
        puzzles_corrected = 0
        puzzles_processed = 0

        # Process each puzzle
        for puzzle_detail in updated_data['details']:
            puzzles_processed += 1
            print(f"Processing puzzle number {puzzles_processed}")
            puzzle_was_incorrect = not puzzle_detail['puzzle_solved']
            puzzle_now_correct = True
            moves_processed = 0

            for move_detail in puzzle_detail['moves']:
                moves_processed += 1
                fen = move_detail['fen']
                correct_move = move_detail['correct_move']
                model_move = move_detail.get('extracted_move') or move_detail.get('engine_move')

                print(f"\nProcessing move number {moves_processed} for puzzle number {puzzles_processed}")
                print(f"  FEN: {fen}")
                print(f"  Correct move: {correct_move}")
                print(f"  Model move: {model_move}")

                # Skip if already correct
                if move_detail['is_correct']:
                    print(f"  → Skipping: already correct")
                    continue

                # Skip if not valid
                if not move_detail.get('is_valid', False):
                    print(f"  → Skipping: invalid move")
                    puzzle_now_correct = False
                    continue

                if not model_move:
                    print(f"Skipping move number {moves_processed} because model move is None")
                    puzzle_now_correct = False
                    continue

                # Check if alternative move achieves same result
                print(f"Checking move number {moves_processed} to see if it is an alternative")
                is_alternative_correct, correct_depth, model_depth, reason = self.check_alternative_move(
                    fen, correct_move, model_move, expected_mate_depth
                )

                if is_alternative_correct:
                    if reason == "same_mate":
                        print(f"   ✓ Alternative mate: {model_move} (M{model_depth}) "
                              f"= {correct_move} (M{correct_depth}) [{fen[:40]}...]")
                    elif reason == "same_eval":
                        print(f"   ✓ Equivalent eval: {model_move} ≈ {correct_move} "
                              f"[{fen[:40]}...]")
                    move_detail['is_correct'] = True
                    move_detail['alternative_solution'] = True
                    move_detail['alternative_reason'] = reason
                    if model_depth is not None:
                        move_detail['mate_depth'] = model_depth
                    moves_corrected += 1
                else:
                    puzzle_now_correct = False
                    if model_depth is not None and correct_depth is not None:
                        print(f"   ✗ Different: {model_move} (M{model_depth}) "
                              f"!= {correct_move} (M{correct_depth}) [{fen[:40]}...]")
                    else:
                        print(f"   ✗ Incorrect: Best move={correct_move} != Model move={model_move}")

            # Update puzzle-level correctness
            if puzzle_was_incorrect and puzzle_now_correct:
                puzzle_detail['puzzle_solved'] = True
                puzzles_corrected += 1

        # Update summary statistics
        updated_data['summary']['moves_correct'] = original_correct + moves_corrected
        updated_data['summary']['puzzles_correct'] = original_puzzles_correct + puzzles_corrected

        # Recalculate percentages
        total_moves = updated_data['summary']['moves_total']
        total_puzzles = updated_data['summary']['puzzles_total']

        if total_moves > 0:
            updated_data['summary']['move_accuracy'] = 100 * updated_data['summary']['moves_correct'] / total_moves
        if total_puzzles > 0:
            updated_data['summary']['puzzle_accuracy'] = 100 * updated_data['summary']['puzzles_correct'] / total_puzzles

        if moves_corrected > 0 or puzzles_corrected > 0:
            print(f"  Adjusted: +{moves_corrected} moves, +{puzzles_corrected} puzzles "
                  f"(Move Acc: {original_correct}/{total_moves} → {updated_data['summary']['moves_correct']}/{total_moves}, "
                  f"Puzzle Acc: {original_puzzles_correct}/{total_puzzles} → {updated_data['summary']['puzzles_correct']}/{total_puzzles})")

        return updated_data

    def process_all_results(self, results_file: str):
        """Process all model results and save adjusted results back to the same file.

        Args:
            results_file: JSON file with model results (will be updated in-place)
        """
        # Load results
        print(f"\nLoading results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        adjusted_results = {}

        # Process each puzzle type
        for puzzle_type in sorted(all_results.keys()):
            print(f"\nProcessing {puzzle_type}...")

            is_mate = self.is_mate_puzzle(puzzle_type)
            if is_mate:
                expected_mate_depth = self.get_expected_mate_depth(puzzle_type)
                print(f"  Mate puzzle - Expected mate depth: {expected_mate_depth}")
            else:
                print(f"  Non-mate puzzle - Checking evaluation equivalence (±20cp)")

            # Check if this is a nested structure (OpenAI) or flat structure (Stockfish/Caissa)
            puzzle_data = all_results[puzzle_type]

            # If puzzle_data has 'summary' and 'details', it's a flat structure
            if isinstance(puzzle_data, dict) and 'summary' in puzzle_data and 'details' in puzzle_data:
                # Flat structure (Stockfish/Caissa)
                model_name = puzzle_data.get('model', puzzle_data.get('engine', 'Unknown'))
                print(f"\n  Checking {model_name}...")

                adjusted_results[puzzle_type] = self.process_model_results(puzzle_data, puzzle_type)
            else:
                # Nested structure (OpenAI)
                adjusted_results[puzzle_type] = {}

                for model_key, model_data in puzzle_data.items():
                    model_name = model_data.get('model', model_key)
                    reasoning = model_data.get('reasoning_effort', '')
                    display_name = f"{model_name} ({reasoning})" if reasoning else model_name

                    print(f"\n  Checking {display_name}...")

                    adjusted_model_data = self.process_model_results(model_data, puzzle_type)
                    adjusted_results[puzzle_type][model_key] = adjusted_model_data

        # Save adjusted results back to the original file
        print(f"\nSaving adjusted results to {results_file}...")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(adjusted_results, f, indent=2)
        print(f"Adjusted results saved!")


def main():
    """Main function to check alternatives and adjust accuracy."""

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Result files to process in-place
    openai_results_file = os.path.join(script_dir, "openai_models_n=100_results.json")
    stockfish_results_file = os.path.join(script_dir, "stockfish17_results.json")

    # Caissa iteration result files
    caissa_results_files = [
        os.path.join(script_dir, "caissa-iters-500k_results.json"),
        os.path.join(script_dir, "caissa-iters-1m_results.json"),
        os.path.join(script_dir, "caissa-iters-1.5m_results.json")
    ]

    # Stockfish path (adjust as needed)
    stockfish_path = r"\the-big-fish\stockfish-windows-x86-64-avx2.exe"

    # Check if Stockfish exists
    if not os.path.exists(stockfish_path):
        print(f"Error: Stockfish not found at {stockfish_path}")
        print("Please update the stockfish_path in the script.")
        return

    print("="*80)
    print("ALTERNATIVE MOVE CHECKER FOR MATE PUZZLES")
    print("="*80)

    # Initialize checker
    checker = AlternativeMoveChecker(stockfish_path)
    checker.start_engine()

    try:
        # Process OpenAI results file (nested structure: puzzle_type -> model_key -> data)
        if os.path.exists(openai_results_file):
            print(f"\n{'='*80}")
            print(f"Processing OpenAI results file: {openai_results_file}")
            print(f"{'='*80}")
            checker.process_all_results(openai_results_file)
        else:
            print(f"Warning: {openai_results_file} not found, skipping...")

        # Process Stockfish results file (flat structure: puzzle_type -> data)
        if os.path.exists(stockfish_results_file):
            print(f"\n{'='*80}")
            print(f"Processing Stockfish results file: {stockfish_results_file}")
            print(f"{'='*80}")
            checker.process_all_results(stockfish_results_file)
        else:
            print(f"Warning: {stockfish_results_file} not found, skipping...")

        # Process Caissa iteration results files (flat structure: puzzle_type -> data)
        for caissa_results_file in caissa_results_files:
            if os.path.exists(caissa_results_file):
                print(f"\n{'='*80}")
                print(f"Processing Caissa results file: {caissa_results_file}")
                print(f"{'='*80}")
                checker.process_all_results(caissa_results_file)
            else:
                print(f"Warning: {caissa_results_file} not found, skipping...")

        print(f"\n{'='*80}")
        print(f"All files have been updated in-place!")
        print(f"{'='*80}")

    finally:
        checker.stop_engine()


if __name__ == "__main__":
    main()
