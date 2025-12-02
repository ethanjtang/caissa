import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import chess
from typing import List, Dict, Optional, Tuple
import json
import time

'''
Test Caissa checkpoints on the same set used to evaluate the GPTs and Stockfish.
Uses the same puzzles as tested in openai_models_results.json for fair comparison.
'''

class GPTChessTester:
    def __init__(self, checkpoint_path: str, use_cpu: bool = False):
        """Initialize the GPT chess model tester."""
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.encode = None
        self.decode = None
        # Use CUDA if available, with float16 for GTX 1060 compatibility (no bfloat16 support)
        self.device = 'cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        # GTX 1060 doesn't support bfloat16, use float16 instead
        if use_cpu:
            self.dtype = 'float32'
        elif torch.cuda.is_available():
            self.dtype = 'float16'  # GTX 1060 compatible
        else:
            self.dtype = 'float32'
        self.ctx = None

    def load_model(self):
        """Load the GPT model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")
        print(f"Using device: {self.device}")

        # Set up dtype
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)

        # Load state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.to(self.device)

        # Set up encoder/decoder
        # First try to load from the same directory as the checkpoint
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        meta_path = os.path.join(checkpoint_dir, 'meta_final.pkl')

        # The below if-else statement represents all the bad things about this world:
        # Desperation, complacency, etc.
        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Loaded character-level tokenizer with vocab size: {len(stoi)}")
        # Then try current directory
        elif os.path.exists('meta_final.pkl'):
            print(f"Loading meta from current directory...")
            with open('meta_final.pkl', 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Loaded character-level tokenizer with vocab size: {len(stoi)}")
        elif 'config' in checkpoint and 'dataset' in checkpoint['config']:
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta_final.pkl')
            if os.path.exists(meta_path):
                print(f"Loading meta from {meta_path}...")
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                stoi, itos = meta['stoi'], meta['itos']
                self.encode = lambda s: [stoi[c] for c in s]
                self.decode = lambda l: ''.join([itos[i] for i in l])
                print(f"Loaded character-level tokenizer with vocab size: {len(stoi)}")
            else:
                print("No meta_final.pkl found, assuming GPT-2 encodings...")
                enc = tiktoken.get_encoding("gpt2")
                self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
                self.decode = lambda l: enc.decode(l)
        else:
            print("No meta_final.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)

        print("Model loaded successfully!")

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

    def generate_move(self, fen: str, max_new_tokens: int = 10, temperature: float = 0.8, top_k: int = 1) -> tuple:
        """Generate a move for the given FEN position.

        Returns:
            tuple: (generated_output, response_time)
        """
        # Create prompt with FEN and Best move labels
        prompt = f"FEN: {fen}\nBest move:"

        # Encode
        start_ids = self.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # Generate (use temperature > 0 to avoid division by zero in multinomial sampling)
        # For deterministic results, use temperature=0.01 and top_k=1
        start_time = time.time()
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, max_new_tokens, temperature=max(temperature, 0.01), top_k=top_k)
                output = self.decode(y[0].tolist())
        response_time = time.time() - start_time

        # Extract just the generated part (after the prompt)
        generated = output[len(prompt):].strip()

        return generated, response_time

    def extract_move_from_output(self, output: str, fen: str) -> Optional[str]:
        """Extract a valid UCI move from the model output."""
        # Try to find a valid UCI move in the output
        board = chess.Board(fen)

        # Check if output contains "Best move:" and extract what follows
        if "Best move:" in output:
            # Find everything after "Best move:"
            after_label = output.split("Best move:", 1)[1].strip()
            # Get the first word after the label
            tokens = after_label.split()
            if tokens:
                candidate = tokens[0].strip().lower().rstrip('.,;:!?()[]{}')
                try:
                    move = chess.Move.from_uci(candidate)
                    if move in board.legal_moves:
                        return move.uci()
                except:
                    pass

        # Also try to extract 4-5 character sequences (UCI moves are typically 4-5 chars like "e2e4" or "e7e8q")
        import re
        # Find all sequences that look like UCI moves (letter+digit+letter+digit, optionally followed by promotion piece)
        uci_patterns = re.findall(r'[a-h][1-8][a-h][1-8][qrbn]?', output.lower())

        # Try UCI pattern matches
        for candidate in uci_patterns:
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        # Then try all tokens
        tokens = output.split()
        for token in tokens:
            # Clean the token
            cleaned = token.strip().lower()
            cleaned = cleaned.rstrip('.,;:!?()[]{}')

            # Try to parse as UCI move
            try:
                move = chess.Move.from_uci(cleaned)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        # If no valid move found, return None
        return None

    def test_puzzles_on_sequences(self, puzzles: List[List[Dict]]) -> Dict:
        """Test the model on puzzle sequences (same format as OpenAI tests).

        Args:
            puzzles: List of puzzles, where each puzzle is a list of positions (moves)

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

        print(f"\nTesting Caissa on {total_puzzles} puzzles ({total_moves} total moves)...")

        for puzzle_idx, puzzle in enumerate(puzzles):
            if (puzzle_idx + 1) % 10 == 0:
                print(f"  Progress: {puzzle_idx + 1}/{total_puzzles} puzzles")

            puzzle_all_correct = True
            puzzle_details = []

            for move_idx, position in enumerate(puzzle):
                fen = position['fen']
                correct_move = position['best_move']

                # Generate move (use very low temperature for deterministic results)
                # number of tokens = 20
                output, response_time = self.generate_move(fen, max_new_tokens=20, temperature=0.01, top_k=1)
                extracted_move = self.extract_move_from_output(output, fen)

                # Accumulate response time
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
                print(f"  {status} Puzzle {puzzle_idx+1} Move {move_idx+1}/{len(puzzle)}: Correct={correct_move}, Model={output[:20]}..., Extracted={extracted_move} [Time: {response_time:.2f}s]")

                puzzle_details.append({
                    'fen': fen,
                    'correct_move': correct_move,
                    'model_output': output,
                    'extracted_move': extracted_move,
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
        print(f"Results for Caissa on {puzzle_type}")
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

    def save_results(self, all_results: Dict, output_path: str):
        """Save detailed results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Detailed results saved to {output_path}")


def test_single_checkpoint(checkpoint_name: str, checkpoint_path: str, puzzles_by_type: Dict,
                           script_dir: str, use_cpu: bool = False) -> Optional[Dict]:
    """Test a single Caissa checkpoint on all puzzle types.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., 'caissa-iters-500k')
        checkpoint_path: Full path to checkpoint file
        puzzles_by_type: Dictionary of puzzles organized by type
        script_dir: Script directory path
        use_cpu: Whether to use CPU instead of GPU

    Returns:
        Dictionary with all results for this checkpoint, or None if checkpoint not found
    """
    print(f"\n{'='*120}")
    print(f"TESTING CHECKPOINT: {checkpoint_name}")
    print(f"{'='*120}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None

    # Initialize tester with CUDA (GTX 1060 compatible with float16)
    tester = GPTChessTester(checkpoint_path, use_cpu=use_cpu)
    tester.load_model()

    # Test on each puzzle set
    all_results = {}

    for puzzle_type, puzzles in puzzles_by_type.items():
        print(f"\n{'='*80}")
        print(f"Testing {puzzle_type} puzzles")
        print(f"{'='*80}")

        total_moves = sum(len(p) for p in puzzles)
        print(f"Testing {len(puzzles)} puzzles ({total_moves} total moves)")

        # Test puzzles
        results = tester.test_puzzles_on_sequences(puzzles)

        # Print results
        tester.print_results(results, puzzle_type)

        # Store results (flat structure compatible with check_for_alternatives.py and compare_all_models.py)
        all_results[puzzle_type] = {
            'model': checkpoint_name,
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
                'avg_response_time_per_puzzle': results['total_response_time'] / results['puzzles_total'] if results['puzzles_total'] > 0 else 0,
                'total_tokens': 0  # Not applicable for Caissa, but included for compatibility
            },
            'details': results['details']
        }

    # Save results for this checkpoint
    output_file = os.path.join(script_dir, f"{checkpoint_name}_results.json")
    tester.save_results(all_results, output_file)

    # Print summary for this checkpoint
    print(f"\n{'='*120}")
    print(f"SUMMARY - {checkpoint_name} on All Puzzle Types")
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

    return all_results


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Define checkpoints to test
    checkpoints = [
        ('caissa-iters-500k', os.path.join(script_dir, "models", "caissa_iters_500k.pt")),
        ('caissa-iters-1m', os.path.join(script_dir, "models", "caissa_iters_1m.pt")),
        ('caissa-iters-1.5m', os.path.join(script_dir, "models", "caissa_iters_1.5m.pt"))
    ]

    # Path to OpenAI results file for loading puzzles
    openai_results_file = os.path.join(project_root, "openai_models_n=100_results.json")

    if not os.path.exists(openai_results_file):
        print(f"Error: OpenAI results file not found at {openai_results_file}")
        print("Please run test_openai_models_reasoning.py first to generate the comparison baseline.")
        return

    # Load puzzles once (will be used for all checkpoints)
    print(f"\nLoading puzzles from {openai_results_file}...")
    # Use a temporary tester just to load puzzles
    temp_checkpoint = checkpoints[0][1]  # Use first checkpoint to load puzzles
    if not os.path.exists(temp_checkpoint):
        print(f"Error: No checkpoint found at {temp_checkpoint}")
        print("Please ensure checkpoint files exist in compare-diff-caissa-iters/caissa/")
        return

    temp_tester = GPTChessTester(temp_checkpoint, use_cpu=False)
    puzzles_by_type = temp_tester.load_puzzles_from_openai_results(openai_results_file)
    print(f"Loaded {len(puzzles_by_type)} puzzle types")

    # Test each checkpoint
    all_checkpoint_results = {}

    for checkpoint_name, checkpoint_path in checkpoints:
        results = test_single_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            puzzles_by_type=puzzles_by_type,
            script_dir=script_dir,
            use_cpu=False  # Set to True if you want to use CPU
        )

        if results:
            all_checkpoint_results[checkpoint_name] = results

    # Print final comparison across all checkpoints
    if all_checkpoint_results:
        print(f"\n{'='*160}")
        print("FINAL COMPARISON - ALL CAISSA CHECKPOINTS")
        print(f"{'='*160}")
        print(f"{'Checkpoint':<20} {'Puzzle Type':<20} {'Move Acc':<20} {'Puzzle Acc':<15} {'Avg Time/Move':<15}")
        print(f"{'-'*160}")

        for checkpoint_name in [cp[0] for cp in checkpoints]:
            if checkpoint_name not in all_checkpoint_results:
                continue

            results = all_checkpoint_results[checkpoint_name]
            for puzzle_type in sorted(results.keys()):
                summary = results[puzzle_type]['summary']
                move_acc = f"{summary['move_accuracy']:>6.2f}% ({summary['moves_correct']}/{summary['moves_total']})"
                puzzle_acc = f"{summary['puzzle_accuracy']:>6.2f}% ({summary['puzzles_correct']}/{summary['puzzles_total']})"
                avg_time = f"{summary['avg_response_time_per_move']:.2f}s"

                print(f"{checkpoint_name:<20} {puzzle_type:<20} {move_acc:<20} {puzzle_acc:<15} {avg_time:<15}")
            print(f"{'-'*160}")

        print(f"{'='*160}\n")

        print("Results saved:")
        for checkpoint_name in all_checkpoint_results.keys():
            output_file = os.path.join(script_dir, f"{checkpoint_name}_results.json")
            print(f"  - {output_file}")


if __name__ == "__main__":
    main()
