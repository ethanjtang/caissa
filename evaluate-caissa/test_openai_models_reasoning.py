import os
import chess
from typing import List, Dict, Optional
import json
from openai import OpenAI
from openai.types.shared_params import Reasoning
from dotenv import load_dotenv
import time
import random

'''
Use to test a range of OpenAI models using their API on chess puzzles extracted from directory .\create-train-val-puzzle-sets
'''

# Load environment variables
load_dotenv()

class OpenAIChessTester:
    def __init__(self, api_key: str):
        """Initialize the OpenAI chess tester."""
        self.client = OpenAI(api_key=api_key)

    def load_puzzles_from_file(self, file_path: str, max_puzzles: int = None, random_sample: bool = True) -> List[List[Dict]]:
        """Load puzzles from validation file, grouped by puzzle.

        Format:
        <puzzle-start/>
        FEN: <fen_string>
        Legal moves: <space_separated_moves>
        Best move: <uci_move>

        <puzzle-end/>

        Returns a list of puzzles, where each puzzle is a list of positions (moves).
        """
        # Legal moves: line may or may not be omitted
        all_puzzles = []
        current_puzzle = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == '<puzzle-start/>':
                current_puzzle = []
            elif line == '<puzzle-end/>':
                if current_puzzle:
                    all_puzzles.append(current_puzzle)
                current_puzzle = []
            elif line.startswith('FEN:'):
                fen = line.replace('FEN:', '').strip()
                legal_moves = None
                best_move = None

                # Look for legal moves and best move in next few lines
                # Check if next line is Legal moves or Best move
                next_line_idx = i + 1
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if next_line.startswith('Legal moves:'):
                        legal_moves = next_line.replace('Legal moves:', '').strip()
                        next_line_idx += 1

                # Check for Best move
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if next_line.startswith('Best move:'):
                        best_move = next_line.replace('Best move:', '').strip()

                if fen and best_move:
                    current_puzzle.append({
                        'fen': fen,
                        'legal_moves': legal_moves.split() if legal_moves else [],
                        'best_move': best_move
                    })

                # Skip ahead to avoid reprocessing
                i = next_line_idx

            i += 1

        # Random sampling if requested
        if max_puzzles and len(all_puzzles) > max_puzzles:
            if random_sample:
                all_puzzles = random.sample(all_puzzles, max_puzzles)
            else:
                all_puzzles = all_puzzles[:max_puzzles]

        return all_puzzles

    def create_prompt(self, fen: str) -> str:
        """Create prompt for the LLM."""
        board = chess.Board(fen)
        side = "White" if board.turn == chess.WHITE else "Black"

        prompt = f"""Given this chess position in FEN notation: {fen}

        Find the best move for {side}. Respond with only the move in UCI format (like e2e4 or g1f3).

        Move:"""

        return prompt
    def query_model(self, model: str, prompt: str, reasoning_effort: str = "minimal", max_retries: int = 3) -> tuple:
        """Query a model and return the response and token usage.

        Returns:
            tuple: (response_text, input_tokens, output_tokens, reasoning_tokens, response_time)
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                if model.startswith("gpt-5"):
                    # Give high/max reasoning much more tokens to think
                    if reasoning_effort in ["high", "medium"]:
                        max_output_tokens = 10000  # Much larger token budget for extended reasoning
                    else:
                        max_output_tokens = 30  # Minimal tokens for quick responses

                    response = self.client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": "You are a chess engine that responds with moves in UCI format only."},
                            {"role": "user", "content": prompt}
                        ],
                        max_output_tokens=max_output_tokens,
                        reasoning=Reasoning(effort=reasoning_effort)
                    )
                    move = response.output_text

                    # Extract token usage
                    input_tokens = response.usage.input_tokens if (hasattr(response, 'usage') and response.usage) else 0
                    output_tokens = response.usage.output_tokens if (hasattr(response, 'usage') and response.usage) else 0
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0) if (hasattr(response, 'usage') and response.usage) else 0

                else:
                    # Older models use chat.completions
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a chess engine that responds with moves in UCI format only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=1000
                    )
                    move = response.choices[0].message.content

                    # Extract token usage
                    input_tokens = response.usage.prompt_tokens if response.usage else 0
                    output_tokens = response.usage.completion_tokens if response.usage else 0
                    reasoning_tokens = 0

                response_time = time.time() - start_time

                if move:
                    return (move.strip(), input_tokens, output_tokens, reasoning_tokens, response_time)
                else:
                    print(f"    Warning: Empty response from {model}")
                    return ("", 0, 0, 0, response_time)

            except Exception as e:
                print(f"    Error querying {model} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return ("", 0, 0, 0, 0.0)

        return ("", 0, 0, 0, 0.0)

    def extract_move_from_output(self, output: str, fen: str) -> Optional[str]:
        """Extract a valid UCI move from the model output."""
        board = chess.Board(fen)

        # Clean the output
        output = output.strip()

        # Try to extract 4-5 character sequences (UCI moves)
        import re
        uci_patterns = re.findall(r'[a-h][1-8][a-h][1-8][qrbn]?', output.lower())

        # Try UCI pattern matches
        for candidate in uci_patterns:
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        # Try all tokens (both UCI and algebraic notation)
        tokens = output.split()
        for token in tokens:
            cleaned = token.strip().rstrip('.,;:!?()[]{}')

            # Try UCI format (lowercase)
            try:
                move = chess.Move.from_uci(cleaned.lower())
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

            # Try algebraic (SAN) notation (case-sensitive)
            try:
                move = board.parse_san(cleaned)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        return None

    def test_model_on_puzzles(self, model: str, puzzles: List[List[Dict]], reasoning_effort: str = "minimal") -> Dict:
        """Test a single model on puzzles.

        Args:
            model: Model name to test
            puzzles: List of puzzles, where each puzzle is a list of positions
            reasoning_effort: Reasoning effort level for GPT-5 models ("minimal" or "max")

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
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_reasoning_tokens': 0,
            'total_response_time': 0.0,
            'details': []
        }

        reasoning_suffix = f" (reasoning={reasoning_effort})" if model.startswith("gpt-5") else ""
        print(f"\nTesting {model}{reasoning_suffix} on {total_puzzles} puzzles ({total_moves} total moves)...")

        for puzzle_idx, puzzle in enumerate(puzzles):
            if (puzzle_idx + 1) % 10 == 0:
                print(f"  Progress: {puzzle_idx + 1}/{total_puzzles} puzzles")

            puzzle_all_correct = True
            puzzle_details = []

            for move_idx, position in enumerate(puzzle):
                fen = position['fen']
                correct_move = position['best_move']

                # Create prompt and query model
                prompt = self.create_prompt(fen)
                query_result = self.query_model(model, prompt, reasoning_effort)
                output, input_tokens, output_tokens, reasoning_tokens, response_time = query_result
                extracted_move = self.extract_move_from_output(output, fen)

                # Accumulate token usage and response time
                results['total_input_tokens'] += input_tokens
                results['total_output_tokens'] += output_tokens
                results['total_reasoning_tokens'] += reasoning_tokens
                results['total_response_time'] += response_time

                # Check validity
                board = chess.Board(fen)
                is_valid = False
                is_correct = False

                if extracted_move:
                    try:
                        move = chess.Move.from_uci(extracted_move)
                        is_valid = move in board.legal_moves
                    except:
                        is_valid = False

                    is_correct = (extracted_move == correct_move)

                if is_valid:
                    results['moves_valid'] += 1
                if is_correct:
                    results['moves_correct'] += 1
                else:
                    puzzle_all_correct = False

                # Print result
                status = "✓" if is_correct else ("?" if is_valid else "✗")
                token_info = f" [Tokens: in={input_tokens}, out={output_tokens}"
                if reasoning_tokens > 0:
                    token_info += f", reasoning={reasoning_tokens}"
                token_info += f", Time: {response_time:.2f}s]"

                print(f"  {status} Puzzle {puzzle_idx+1} Move {move_idx+1}/{len(puzzle)}: Correct={correct_move}{token_info}")
                print(f"      Model Response: '{output}'")
                print(f"      Extracted={extracted_move}")

                puzzle_details.append({
                    'fen': fen,
                    'correct_move': correct_move,
                    'model_output': output,
                    'extracted_move': extracted_move,
                    'is_valid': is_valid,
                    'is_correct': is_correct,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'reasoning_tokens': reasoning_tokens,
                    'response_time': response_time
                })

                # Small delay to avoid rate limiting
                time.sleep(0.2)

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

    def print_results(self, results: Dict, model_name: str, puzzle_type: str, reasoning_effort: Optional[str] = None):
        """Print results summary."""
        moves_total = results['moves_total']
        moves_correct = results['moves_correct']
        moves_valid = results['moves_valid']
        puzzles_total = results['puzzles_total']
        puzzles_correct = results['puzzles_correct']
        total_input_tokens = results.get('total_input_tokens', 0)
        total_output_tokens = results.get('total_output_tokens', 0)
        total_reasoning_tokens = results.get('total_reasoning_tokens', 0)
        total_response_time = results.get('total_response_time', 0.0)

        reasoning_suffix = f" (reasoning={reasoning_effort})" if reasoning_effort else ""

        print(f"\n{'='*80}")
        print(f"Results for {model_name}{reasoning_suffix} on {puzzle_type}")
        print(f"{'='*80}")
        print(f"MOVE-BY-MOVE ACCURACY:")
        print(f"  Total moves: {moves_total}")
        if moves_total > 0:
            print(f"  Correct moves: {moves_correct} ({100*moves_correct/moves_total:.2f}%)")
            print(f"  Valid moves: {moves_valid} ({100*moves_valid/moves_total:.2f}%)")
        else:
            print(f"  Correct moves: {moves_correct} (N/A - no moves loaded)")
            print(f"  Valid moves: {moves_valid} (N/A - no moves loaded)")
        print(f"\nPUZZLE-LEVEL ACCURACY:")
        print(f"  Total puzzles: {puzzles_total}")
        if puzzles_total > 0:
            print(f"  Fully solved puzzles: {puzzles_correct} ({100*puzzles_correct/puzzles_total:.2f}%)")
        else:
            print(f"  Fully solved puzzles: {puzzles_correct} (N/A - no puzzles loaded)")
        print(f"\nTOKEN USAGE:")
        print(f"  Input tokens: {total_input_tokens:,}")
        print(f"  Output tokens: {total_output_tokens:,}")
        if total_reasoning_tokens > 0:
            print(f"  Reasoning tokens: {total_reasoning_tokens:,}")
        print(f"  Total tokens: {total_input_tokens + total_output_tokens + total_reasoning_tokens:,}")
        if moves_total > 0:
            avg_tokens = (total_input_tokens + total_output_tokens + total_reasoning_tokens) / moves_total
            print(f"  Average tokens per move: {avg_tokens:.1f}")
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
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Go up one level to the project root, then into puzzles/validation_puzzles
    validation_dir = os.path.join(project_root, "puzzles", "validation_puzzles")

    # Models to test - GPT-5 series with reasoning levels and GPT-3.5-turbo
    models_and_reasoning = [
        # ('gpt-5', 'high'),                # GPT-5 with high reasoning
        # ('gpt-5', 'medium'),                # GPT-5 with medium reasoning
        ('gpt-5', 'minimal'),            # GPT-5 with minimal reasoning
        ('gpt-5-mini', 'minimal'),       # GPT-5-mini with minimal reasoning
        ('gpt-5-nano', 'minimal'),       # GPT-5-nano with minimal reasoning
        ('gpt-3.5-turbo', None)          # GPT-3.5-turbo (no reasoning parameter)
    ]

    # Puzzle files to test (100 puzzles each)
    puzzle_files = {
        'mate_in_1': 'validation_puzzles_mateIn1.txt',
        'mate_in_2': 'validation_puzzles_mateIn2.txt',
        'mate_in_3': 'validation_puzzles_mateIn3.txt',
        'one_move': 'validation_puzzles_oneMove.txt',
        'middlegame': 'validation_puzzles_middlegame.txt',
        'endgame': 'validation_puzzles_endgame.txt',
        'zugzwang': 'validation_puzzles_zugzwang.txt',
        'crushing': 'validation_puzzles_crushing.txt',
        'master_vs_master': 'validation_puzzles_masterVsMaster.txt',
        'superGM': 'validation_puzzles_superGM.txt'
    }

    # Initialize tester
    tester = OpenAIChessTester(api_key)

    # Store all results
    all_results = {}

    # Test each puzzle type
    for puzzle_type, filename in puzzle_files.items():
        file_path = os.path.join(validation_dir, filename)

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Loading {puzzle_type} puzzles from {filename}")
        print(f"{'='*80}")

        # Load N puzzles randomly sampled
        puzzles = tester.load_puzzles_from_file(file_path, max_puzzles=100, random_sample=True)
        total_moves = sum(len(p) for p in puzzles)
        print(f"Loaded {len(puzzles)} puzzles ({total_moves} total moves)")

        # Test each model with different reasoning levels
        all_results[puzzle_type] = {}

        for model, reasoning_effort in models_and_reasoning:
            # Create a unique key for this model+reasoning combination
            if reasoning_effort:
                model_key = f"{model}_{reasoning_effort}"
            else:
                model_key = model
                reasoning_effort = "minimal"  # Default for non-GPT-5 models

            results = tester.test_model_on_puzzles(model, puzzles, reasoning_effort or "minimal")
            tester.print_results(results, model, puzzle_type, reasoning_effort if (reasoning_effort and model.startswith("gpt-5")) else None)

            # Store results
            all_results[puzzle_type][model_key] = {
                'model': model,
                'reasoning_effort': reasoning_effort if model.startswith("gpt-5") else None,
                'summary': {
                    'moves_total': results['moves_total'],
                    'moves_correct': results['moves_correct'],
                    'moves_valid': results['moves_valid'],
                    'move_accuracy': 100 * results['moves_correct'] / results['moves_total'],
                    'move_valid_rate': 100 * results['moves_valid'] / results['moves_total'],
                    'puzzles_total': results['puzzles_total'],
                    'puzzles_correct': results['puzzles_correct'],
                    'puzzle_accuracy': 100 * results['puzzles_correct'] / results['puzzles_total'],
                    'total_input_tokens': results['total_input_tokens'],
                    'total_output_tokens': results['total_output_tokens'],
                    'total_reasoning_tokens': results['total_reasoning_tokens'],
                    'total_tokens': results['total_input_tokens'] + results['total_output_tokens'] + results['total_reasoning_tokens'],
                    'total_response_time': results['total_response_time'],
                    'avg_response_time_per_move': results['total_response_time'] / results['moves_total'] if results['moves_total'] > 0 else 0,
                    'avg_response_time_per_puzzle': results['total_response_time'] / results['puzzles_total'] if results['puzzles_total'] > 0 else 0
                },
                'details': results['details']
            }

    # Save all results to project root directory
    output_file = os.path.join(project_root, "openai_models_n=100_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    # Print summary by theme
    print(f"\n{'='*140}")
    print("SUMMARY - By Puzzle Type (Theme)")
    print(f"{'='*140}")
    print(f"{'Puzzle Type':<20} {'Model':<30} {'Move Acc':<20} {'Puzzle Acc':<15} {'Sanity':<15} {'Avg Time/Move':<15} {'Tokens':<15}")
    print(f"{'-'*140}")

    for puzzle_type in puzzle_files.keys():
        if puzzle_type not in all_results:
            continue
        for model_key in sorted(all_results[puzzle_type].keys()):
            result_data = all_results[puzzle_type][model_key]
            summary = result_data['summary']
            model_name = result_data['model']
            reasoning = result_data.get('reasoning_effort')

            # Create display name with reasoning level
            if reasoning:
                display_name = f"{model_name} ({reasoning})"
            else:
                display_name = model_name

            # Calculate sanity (valid move rate)
            sanity = summary['move_valid_rate']
            sanity_str = f"{sanity:>6.2f}%"

            move_acc = f"{summary['move_accuracy']:>6.2f}% ({summary['moves_correct']}/{summary['moves_total']})"
            puzzle_acc = f"{summary['puzzle_accuracy']:>6.2f}% ({summary['puzzles_correct']}/{summary['puzzles_total']})"
            avg_time = f"{summary['avg_response_time_per_move']:.2f}s"
            total_tokens = f"{summary['total_tokens']:,}"

            print(f"{puzzle_type:<20} {display_name:<30} {move_acc:<20} {puzzle_acc:<15} {sanity_str:<15} {avg_time:<15} {total_tokens}")

    print(f"{'='*140}\n")

    # Print overall model summary
    print(f"\n{'='*135}")
    print("OVERALL MODEL PERFORMANCE - Aggregated Across All Themes")
    print(f"{'='*135}")
    print(f"{'Model':<35} {'Move Acc':<20} {'Puzzle Acc':<20} {'Sanity':<15} {'Avg Time/Move':<15} {'Total Tokens':<15}")
    print(f"{'-'*135}")

    # Aggregate results by model
    model_aggregates = {}
    for puzzle_type in all_results.keys():
        for model_key, result_data in all_results[puzzle_type].items():
            if model_key not in model_aggregates:
                model_aggregates[model_key] = {
                    'total_moves': 0,
                    'total_correct_moves': 0,
                    'total_valid_moves': 0,
                    'total_puzzles': 0,
                    'total_correct_puzzles': 0,
                    'total_time': 0.0,
                    'total_tokens': 0,
                    'model_name': result_data['model'],
                    'reasoning': result_data.get('reasoning_effort')
                }

            summary = result_data['summary']
            model_aggregates[model_key]['total_moves'] += summary['moves_total']
            model_aggregates[model_key]['total_correct_moves'] += summary['moves_correct']
            model_aggregates[model_key]['total_valid_moves'] += summary['moves_valid']
            model_aggregates[model_key]['total_puzzles'] += summary['puzzles_total']
            model_aggregates[model_key]['total_correct_puzzles'] += summary['puzzles_correct']
            model_aggregates[model_key]['total_time'] += summary['total_response_time']
            model_aggregates[model_key]['total_tokens'] += summary['total_tokens']

    # Print aggregated results
    for model_key in sorted(model_aggregates.keys()):
        agg = model_aggregates[model_key]

        # Create display name with reasoning level
        if agg['reasoning']:
            display_name = f"{agg['model_name']} ({agg['reasoning']})"
        else:
            display_name = agg['model_name']

        # Calculate aggregated metrics
        move_acc_pct = (100 * agg['total_correct_moves'] / agg['total_moves']) if agg['total_moves'] > 0 else 0
        puzzle_acc_pct = (100 * agg['total_correct_puzzles'] / agg['total_puzzles']) if agg['total_puzzles'] > 0 else 0
        sanity_pct = (100 * agg['total_valid_moves'] / agg['total_moves']) if agg['total_moves'] > 0 else 0
        avg_time = (agg['total_time'] / agg['total_moves']) if agg['total_moves'] > 0 else 0

        move_acc = f"{move_acc_pct:>6.2f}% ({agg['total_correct_moves']}/{agg['total_moves']})"
        puzzle_acc = f"{puzzle_acc_pct:>6.2f}% ({agg['total_correct_puzzles']}/{agg['total_puzzles']})"
        sanity_str = f"{sanity_pct:>6.2f}%"
        avg_time_str = f"{avg_time:.2f}s"
        total_tokens_str = f"{agg['total_tokens']:,}"

        print(f"{display_name:<35} {move_acc:<20} {puzzle_acc:<20} {sanity_str:<15} {avg_time_str:<15} {total_tokens_str:<15}")

    print(f"{'='*135}\n")


if __name__ == "__main__":
    main()
