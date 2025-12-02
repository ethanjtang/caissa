import os
import json
import time
import chess
import chess.engine
import torch
import pickle
from contextlib import nullcontext
from typing import Dict, List, Optional
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

'''
Test rapid instances of Stockfish 17 and Caissa models on speed benchmarks.
Uses puzzles from openai_models_speeds.json and generate response time comparison graphics.
'''

# Import Caissa model components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import GPTConfig, GPT


class StockfishTester:
    def __init__(self, engine_path: str, think_time: float = 1.0):
        self.engine_path = engine_path
        self.think_time = think_time
        self.engine = None

    def start_engine(self):
        print(f"Starting Stockfish 17 (think_time={self.think_time}s)...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

    def stop_engine(self):
        if self.engine:
            self.engine.quit()

    def query_engine(self, fen: str) -> tuple:
        board = chess.Board(fen)
        start_time = time.time()
        result = self.engine.play(board, chess.engine.Limit(time=self.think_time))
        response_time = time.time() - start_time
        return (result.move.uci() if result.move else None, response_time)


class CaissaTester:
    def __init__(self, checkpoint_path: str, model_name: str):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.model = None
        self.encode = None
        self.decode = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'float16' if torch.cuda.is_available() else 'float32'
        self.ctx = None

    def load_model(self):
        print(f"Loading {self.model_name} from {self.checkpoint_path}...")
        ptdtype = {'float32': torch.float32, 'float16': torch.float16}[self.dtype]
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # Load tokenizer - try multiple locations like test_all_caissa_ckpts.py
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        meta_path = os.path.join(checkpoint_dir, 'meta_final.pkl')

        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Loaded character-level tokenizer with vocab size: {len(stoi)}")
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
                raise FileNotFoundError(f"Could not find meta_final.pkl in any expected location")
        else:
            raise FileNotFoundError(f"Could not find meta_final.pkl in any expected location")

        print(f"{self.model_name} loaded successfully!")

    def query_model(self, fen: str, max_new_tokens: int = 10) -> tuple:
        """Query the Caissa model for a move given a FEN position.

        Returns:
            tuple: (move_uci, response_time)
        """
        # Use the same prompt format as test_all_caissa_ckpts.py
        prompt = f"FEN: {fen}\nBest move:"
        start_ids = self.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # Generate with deterministic settings (low temperature, top_k=1)
        start_time = time.time()
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, max_new_tokens, temperature=0.01, top_k=1)
                output = self.decode(y[0].tolist())
        response_time = time.time() - start_time

        # Extract just the generated part (after the prompt)
        generated = output[len(prompt):].strip()

        # Extract valid UCI move from output
        move = self._extract_move_from_output(generated, fen)

        return (move if move else "", response_time)

    def _extract_move_from_output(self, output: str, fen: str) -> Optional[str]:
        """Extract a valid UCI move from the model output."""
        import re

        try:
            board = chess.Board(fen)
        except:
            return None

        # Check if output contains "Best move:" and extract what follows
        if "Best move:" in output:
            after_label = output.split("Best move:", 1)[1].strip()
            tokens = after_label.split()
            if tokens:
                candidate = tokens[0].strip().lower().rstrip('.,;:!?()[]{}')
                try:
                    move = chess.Move.from_uci(candidate)
                    if move in board.legal_moves:
                        return move.uci()
                except:
                    pass

        # Find all sequences that look like UCI moves
        uci_patterns = re.findall(r'[a-h][1-8][a-h][1-8][qrbn]?', output.lower())
        for candidate in uci_patterns:
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        # Try all tokens
        tokens = output.split()
        for token in tokens:
            cleaned = token.strip().lower().rstrip('.,;:!?()[]{}')
            try:
                move = chess.Move.from_uci(cleaned)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        return None


class AlternativeMoveChecker:
    """Check if alternative moves are equivalent to correct moves."""

    def __init__(self, engine: chess.engine.SimpleEngine):
        self.engine = engine

    def check_alternative_move(self, fen: str, correct_move: str, model_move: str) -> tuple:
        """Check if model's move is equivalent to correct move.

        Returns:
            Tuple of (is_alternative_correct, reason)
        """
        if model_move == correct_move:
            return (True, "exact")

        # Check mate depth
        correct_mate_depth = self.get_mate_depth(fen, correct_move)
        model_mate_depth = self.get_mate_depth(fen, model_move)

        if correct_mate_depth is not None and model_mate_depth is not None:
            if model_mate_depth == correct_mate_depth:
                return (True, "same_mate")
            return (False, "different")

        if correct_mate_depth is not None or model_mate_depth is not None:
            return (False, "different")

        # Check evaluation
        correct_eval = self.get_move_evaluation(fen, correct_move)
        model_eval = self.get_move_evaluation(fen, model_move)

        if correct_eval is not None and model_eval is not None:
            eval_diff = abs(correct_eval - model_eval)
            if eval_diff <= 20.0:
                return (True, "same_eval")

        return (False, "different")

    def get_mate_depth(self, fen: str, move_uci: str) -> Optional[int]:
        """Get mate depth after making a move."""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                return None

            board.push(move)

            if board.is_checkmate():
                return 1

            info = self.engine.analyse(board, chess.engine.Limit(depth=20))
            score = info.get('score')

            if score:
                relative_score = score.relative
                if relative_score.is_mate():
                    mate_value = relative_score.mate()
                    if mate_value is not None and mate_value < 0:
                        return abs(mate_value) + 1
            return None
        except:
            return None

    def get_move_evaluation(self, fen: str, move_uci: str) -> Optional[float]:
        """Get centipawn evaluation after making a move."""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                return None

            board.push(move)

            if board.is_checkmate():
                return 10000.0

            info = self.engine.analyse(board, chess.engine.Limit(depth=20))
            score = info.get('score')

            if score:
                relative_score = score.relative
                if relative_score.is_mate():
                    mate_value = relative_score.mate()
                    if mate_value is not None:
                        if mate_value < 0:
                            return 10000.0 - abs(mate_value)
                        else:
                            return -10000.0 + mate_value
                else:
                    cp_score = relative_score.score()
                    if cp_score is not None:
                        return -cp_score
            return None
        except:
            return None


def load_puzzles_from_speeds_json(speeds_file: str) -> Dict[str, List[Dict]]:
    """Load puzzles from openai_models_speeds.json"""
    with open(speeds_file, 'r', encoding='utf-8') as f:
        speeds_data = json.load(f)

    puzzles_by_type = {}

    for puzzle_type, models_data in speeds_data.items():
        # Get first model's details
        first_model_key = list(models_data.keys())[0]
        details = models_data[first_model_key]['details']

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


def test_model_on_puzzles(tester, puzzles: List[List[Dict]], model_name: str) -> Dict:
    """Test a model on puzzles and return statistics"""
    total_moves = sum(len(puzzle) for puzzle in puzzles)
    total_puzzles = len(puzzles)

    moves_correct = 0
    moves_valid = 0
    puzzles_correct = 0
    total_response_time = 0.0
    details = []

    print(f"Testing {model_name} on {total_puzzles} puzzles ({total_moves} moves)...")
    print()

    for puzzle_idx, puzzle in enumerate(puzzles):
        puzzle_detail = {'puzzle_index': puzzle_idx, 'moves': []}
        puzzle_correct = True

        print(f"  Puzzle #{puzzle_idx + 1}:")
        print(f"  {'-'*100}")

        for move_idx, move_data in enumerate(puzzle):
            fen = move_data['fen']
            correct_move = move_data['best_move']

            model_move, response_time = tester.query_engine(fen) if hasattr(tester, 'query_engine') else tester.query_model(fen)

            total_response_time += response_time

            # Validate move
            is_valid = False
            is_correct = False

            if model_move:
                try:
                    board = chess.Board(fen)
                    move_obj = chess.Move.from_uci(model_move)
                    if move_obj in board.legal_moves:
                        is_valid = True
                        moves_valid += 1
                        if model_move == correct_move:
                            is_correct = True
                            moves_correct += 1
                        else:
                            puzzle_correct = False
                    else:
                        puzzle_correct = False
                except:
                    puzzle_correct = False
            else:
                puzzle_correct = False

            # Print position details
            status = "✓ CORRECT" if is_correct else ("✗ WRONG" if is_valid else "✗ INVALID")
            print(f"    Position {move_idx + 1}:")
            print(f"      FEN:          {fen}")
            print(f"      Correct Move: {correct_move}")
            print(f"      Model Move:   {model_move if model_move else 'None'}")
            print(f"      Status:       {status}")
            print(f"      Time:         {response_time:.4f}s")
            print()

            puzzle_detail['moves'].append({
                'fen': fen,
                'correct_move': correct_move,
                'model_output': model_move,
                'is_valid': is_valid,
                'is_correct': is_correct,
                'response_time': response_time
            })

        if puzzle_correct:
            puzzles_correct += 1
            print(f"  ✓✓✓ Puzzle #{puzzle_idx + 1} SOLVED COMPLETELY ✓✓✓")
        else:
            print(f"  ✗✗✗ Puzzle #{puzzle_idx + 1} NOT SOLVED ✗✗✗")
        print(f"  {'='*100}")
        print()

        details.append(puzzle_detail)

    return {
        'model': model_name,
        'summary': {
            'moves_total': total_moves,
            'moves_correct': moves_correct,
            'moves_valid': moves_valid,
            'move_accuracy': 100 * moves_correct / total_moves if total_moves > 0 else 0,
            'move_valid_rate': 100 * moves_valid / total_moves if total_moves > 0 else 0,
            'puzzles_total': total_puzzles,
            'puzzles_correct': puzzles_correct,
            'puzzle_accuracy': 100 * puzzles_correct / total_puzzles if total_puzzles > 0 else 0,
            'total_response_time': total_response_time,
            'avg_response_time_per_move': total_response_time / total_moves if total_moves > 0 else 0
        },
        'details': details
    }


def calculate_overall_stats(all_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Calculate overall statistics across all puzzle types"""
    model_totals = {}

    for puzzle_type, model_data in all_results.items():
        model_name = model_data['model']

        if model_name not in model_totals:
            model_totals[model_name] = {
                'total_moves': 0,
                'total_moves_correct': 0,
                'total_puzzles': 0,
                'total_puzzles_correct': 0,
                'total_response_time': 0.0
            }

        summary = model_data['summary']
        model_totals[model_name]['total_moves'] += summary['moves_total']
        model_totals[model_name]['total_moves_correct'] += summary['moves_correct']
        model_totals[model_name]['total_puzzles'] += summary['puzzles_total']
        model_totals[model_name]['total_puzzles_correct'] += summary['puzzles_correct']
        model_totals[model_name]['total_response_time'] += summary['total_response_time']

    overall_stats = {}
    for model_name, totals in model_totals.items():
        overall_stats[model_name] = {
            'move_accuracy': 100 * totals['total_moves_correct'] / totals['total_moves'] if totals['total_moves'] > 0 else 0,
            'puzzle_accuracy': 100 * totals['total_puzzles_correct'] / totals['total_puzzles'] if totals['total_puzzles'] > 0 else 0,
            'avg_time_per_move': totals['total_response_time'] / totals['total_moves'] if totals['total_moves'] > 0 else 0
        }

    return overall_stats


def check_alternatives_and_update_results(all_results: Dict[str, Dict], checker: AlternativeMoveChecker) -> Dict[str, Dict]:
    """Check for alternative correct moves and update statistics."""

    updated_results = {}

    for key, model_data in all_results.items():
        # Deep copy to avoid modifying original
        updated_data = deepcopy(model_data)

        # Track updated statistics
        moves_correct_updated = 0
        puzzles_correct_updated = 0

        # Process each puzzle
        for puzzle_detail in updated_data['details']:
            puzzle_all_correct = True

            for move_detail in puzzle_detail['moves']:
                fen = move_detail['fen']
                correct_move = move_detail['correct_move']
                model_move = move_detail['model_output']

                # Check if already correct or invalid
                if move_detail['is_correct']:
                    moves_correct_updated += 1
                    move_detail['alternative_check'] = 'exact'
                elif not move_detail['is_valid'] or not model_move:
                    puzzle_all_correct = False
                    move_detail['alternative_check'] = 'invalid'
                else:
                    # Check for alternative
                    is_alternative, reason = checker.check_alternative_move(fen, correct_move, model_move)
                    move_detail['alternative_check'] = reason

                    if is_alternative:
                        move_detail['is_correct_alternative'] = True
                        moves_correct_updated += 1
                        print(f"  Alternative found ({reason}): FEN={fen[:30]}... Correct={correct_move} Model={model_move}")
                    else:
                        puzzle_all_correct = False
                        move_detail['is_correct_alternative'] = False

            if puzzle_all_correct:
                puzzles_correct_updated += 1

        # Update summary statistics
        total_moves = updated_data['summary']['moves_total']
        total_puzzles = updated_data['summary']['puzzles_total']

        updated_data['summary']['moves_correct'] = moves_correct_updated
        updated_data['summary']['puzzles_correct'] = puzzles_correct_updated
        updated_data['summary']['move_accuracy'] = 100 * moves_correct_updated / total_moves if total_moves > 0 else 0
        updated_data['summary']['puzzle_accuracy'] = 100 * puzzles_correct_updated / total_puzzles if total_puzzles > 0 else 0

        updated_results[key] = updated_data

    return updated_results


def print_detailed_results(all_results: Dict[str, Dict]):
    """Print detailed results for each model's attempt on each puzzle position."""

    print("\n" + "="*150)
    print("DETAILED RESULTS - Each Model's Attempt on Each Puzzle Position")
    print("="*150)

    for key, model_data in all_results.items():
        model_name = model_data['model']
        print(f"\n{'='*150}")
        print(f"MODEL: {model_name}")
        print(f"{'='*150}")

        for puzzle_detail in model_data['details']:
            puzzle_idx = puzzle_detail['puzzle_index']
            print(f"\n  Puzzle #{puzzle_idx + 1}:")
            print(f"  {'-'*140}")

            for move_idx, move_detail in enumerate(puzzle_detail['moves']):
                fen = move_detail['fen']
                correct_move = move_detail['correct_move']
                model_move = move_detail['model_output']
                is_correct = move_detail.get('is_correct', False)
                is_alt = move_detail.get('is_correct_alternative', False)
                alt_check = move_detail.get('alternative_check', 'unknown')
                response_time = move_detail.get('response_time', 0)

                status = "✓ CORRECT" if is_correct else ("✓ ALT" if is_alt else "✗ WRONG")

                print(f"    Position {move_idx + 1}:")
                print(f"      FEN:          {fen}")
                print(f"      Correct Move: {correct_move}")
                print(f"      Model Move:   {model_move}")
                print(f"      Status:       {status} ({alt_check})")
                print(f"      Time:         {response_time:.4f}s")
                print()


def generate_response_time_bar_chart(overall_stats: Dict[str, Dict], output_dir: str):
    """Generate bar chart for model response times with logarithmic scale"""

    # Sort by response time (ascending - faster is better)
    sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['avg_time_per_move'])

    model_names = [name for name, _ in sorted_models]
    response_times = [stats['avg_time_per_move'] for _, stats in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Customize the chart
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Response Time per Move (seconds)', fontsize=16, fontweight='bold')
    ax.set_title('Response Times - SuperGM Theme (n=21 positions)', fontsize=18, fontweight='bold', pad=30)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=13)

    # Use logarithmic scale for y-axis to handle large variations (e.g., GPT-5 medium)
    ax.set_yscale('log')

    # Set up grid for log scale
    ax.grid(axis='y', which='major', alpha=0.6, linestyle='-', linewidth=1.2)
    ax.grid(axis='y', which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=13)

    # Create bar chart with special highlighting
    # GPT-5 medium: red, Caissa: purple, Stockfish: orange, others: blue
    bar_colors = []
    for name in model_names:
        if 'gpt-5' in name.lower() and 'medium' in name.lower():
            bar_colors.append("#b41f1fff")
        elif 'caissa' in name.lower():
            bar_colors.append('#9467bd')  # Purple
        elif 'stockfish' in name.lower():
            bar_colors.append('#ff7f0e')  # Orange
        else:
            bar_colors.append('#1f77b4')  # Blue

    bars = ax.bar(range(len(model_names)), response_times, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, response_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.4f}s',
                ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=4)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'detailed_speeds_response_times.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed response time bar chart to: {output_path}")
    plt.close()


def generate_move_accuracy_bar_chart(overall_stats: Dict[str, Dict], output_dir: str):
    """Generate bar chart for move accuracy (n=21 positions from SuperGM)"""

    # Sort by move accuracy (descending)
    sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['move_accuracy'], reverse=True)

    model_names = [name for name, _ in sorted_models]
    move_accuracies = [stats['move_accuracy'] for _, stats in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Customize the chart
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Move Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Move Accuracy - SuperGM Theme (n=21 positions)', fontsize=18, fontweight='bold', pad=30)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=13)

    # Set up grid with major ticks every 10% and minor ticks every 1%
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', which='major', alpha=0.6, linestyle='-', linewidth=1.2)
    ax.grid(axis='y', which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=13)

    # Create bar chart with special highlighting
    # GPT-5 medium: green, Caissa: purple, Stockfish: orange, others: blue
    bar_colors = []
    for name in model_names:
        if 'gpt-5' in name.lower() and 'medium' in name.lower():
            bar_colors.append('green')
        elif 'caissa' in name.lower():
            bar_colors.append('#9467bd')  # Purple
        elif 'stockfish' in name.lower():
            bar_colors.append('#ff7f0e')  # Orange
        else:
            bar_colors.append('#1f77b4')  # Blue

    bars = ax.bar(range(len(model_names)), move_accuracies, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, move_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=4)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'detailed_speeds_move_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved move accuracy bar chart to: {output_path}")
    plt.close()


def generate_puzzle_accuracy_bar_chart(overall_stats: Dict[str, Dict], output_dir: str):
    """Generate bar chart for puzzle accuracy (n=10 puzzles from SuperGM)"""

    # Sort by puzzle accuracy (descending)
    sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['puzzle_accuracy'], reverse=True)

    model_names = [name for name, _ in sorted_models]
    puzzle_accuracies = [stats['puzzle_accuracy'] for _, stats in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Customize the chart
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Puzzle Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Puzzle Accuracy - SuperGM Theme (n=10 puzzles)', fontsize=18, fontweight='bold', pad=30)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=13)

    # Set up grid with major ticks every 10% and minor ticks every 1%
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', which='major', alpha=0.6, linestyle='-', linewidth=1.2)
    ax.grid(axis='y', which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=13)

    # Create bar chart with special highlighting
    # GPT-5 medium: green, Caissa: purple, Stockfish: orange, others: blue
    bar_colors = []
    for name in model_names:
        if 'gpt-5' in name.lower() and 'medium' in name.lower():
            bar_colors.append('green')
        elif 'caissa' in name.lower():
            bar_colors.append('#9467bd')  # Purple
        elif 'stockfish' in name.lower():
            bar_colors.append('#ff7f0e')  # Orange
        else:
            bar_colors.append('#1f77b4')  # Blue

    bars = ax.bar(range(len(model_names)), puzzle_accuracies, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, puzzle_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=4)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'detailed_speeds_puzzle_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved puzzle accuracy bar chart to: {output_path}")
    plt.close()


def generate_output_tokens_bar_chart(all_results: Dict[str, Dict], output_dir: str):
    """Generate bar chart for average output + reasoning token usage per position (excluding Stockfish 17)"""

    # Calculate average output + reasoning tokens per position for each model
    model_tokens = {}
    model_moves = {}

    for key, model_data in all_results.items():
        model_name = model_data['model']

        # Skip Stockfish
        if 'stockfish' in model_name.lower():
            continue

        # Treat all Caissa models as a single "Caissa" entry
        if 'caissa' in model_name.lower():
            model_name = 'Caissa'

        # Get output and reasoning tokens from summary (if available)
        summary = model_data.get('summary', {})
        output_tokens = summary.get('total_output_tokens', 0)
        reasoning_tokens = summary.get('total_reasoning_tokens', 0)
        total_tokens = output_tokens + reasoning_tokens
        moves_total = summary.get('moves_total', 0)

        # For Caissa models without token data, use 10 tokens per move
        if 'caissa' in model_data['model'].lower() and total_tokens == 0:
            total_tokens = 10 * moves_total

        if model_name not in model_tokens:
            model_tokens[model_name] = 0
            model_moves[model_name] = 0

        model_tokens[model_name] += total_tokens
        model_moves[model_name] += moves_total

    # Calculate average tokens per position
    avg_tokens_per_position = {}
    for model_name in model_tokens:
        if model_moves[model_name] > 0:
            avg_tokens_per_position[model_name] = model_tokens[model_name] / model_moves[model_name]

    # Filter out models with zero tokens
    avg_tokens_per_position = {k: v for k, v in avg_tokens_per_position.items() if v > 0}

    if not avg_tokens_per_position:
        print("No token data available. Skipping output + reasoning tokens chart.")
        return

    # Sort by average tokens (ascending - smallest to largest)
    sorted_models = sorted(avg_tokens_per_position.items(), key=lambda x: x[1])

    model_names = [name for name, _ in sorted_models]
    avg_tokens = [token_count for _, token_count in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Customize the chart
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Output + Reasoning Tokens per Position', fontsize=16, fontweight='bold')
    ax.set_title('Average Output Token Usage per Position - SuperGM Theme (n=21 positions)', fontsize=18, fontweight='bold', pad=30)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=13)

    # Use logarithmic scale for y-axis to handle large variations
    ax.set_yscale('log')

    # Set up grid for log scale
    ax.grid(axis='y', which='major', alpha=0.6, linestyle='-', linewidth=1.2)
    ax.grid(axis='y', which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=13)

    # Create bar chart with special highlighting
    # GPT-5 medium: red, Caissa: purple, others: blue
    bar_colors = []
    for name in model_names:
        if 'gpt-5' in name.lower() and 'medium' in name.lower():
            bar_colors.append("#b41f1fff")
        elif 'caissa' in name.lower():
            bar_colors.append('#9467bd')  # Purple
        else:
            bar_colors.append('#1f77b4')  # Blue

    bars = ax.bar(range(len(model_names)), avg_tokens, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)

    # Add value labels on bars
    for i, (bar, token_count) in enumerate(zip(bars, avg_tokens)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{token_count:.1f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=4)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'detailed_speeds_avg_tokens_per_position.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved average tokens per position bar chart to: {output_path}")
    plt.close()


def main():
    """Main function to test models and generate graphics"""

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    speeds_file = os.path.join(script_dir, 'openai_models_speeds.json')
    output_json_path = os.path.join(script_dir, "SGM_n=10_tests.json")

    # Stockfish path
    stockfish_path = r"C:\Users\ejtan\OneDrive\Desktop\lichess-puzzle-data-for-llms\the-big-fish\stockfish-windows-x86-64-avx2.exe"

    # Caissa checkpoint paths
    caissa_checkpoints = {
        'caissa-iters-500k': r"C:\Users\ejtan\OneDrive\Desktop\lichess-puzzle-data-for-llms\compare-diff-caissa-iters\models\caissa_iters_500k.pt",
        'caissa-iters-1m': r"C:\Users\ejtan\OneDrive\Desktop\lichess-puzzle-data-for-llms\compare-diff-caissa-iters\models\caissa_iters_1m.pt",
        'caissa-iters-1.5m': r"C:\Users\ejtan\OneDrive\Desktop\lichess-puzzle-data-for-llms\compare-diff-caissa-iters\models\caissa_iters_1.5m.pt"
    }

    print("="*100)
    print("DETAILED SPEED COMPARISON - Stockfish 17 and Caissa Models")
    print("="*100)

    # Check if results already exist
    if os.path.exists(output_json_path):
        print(f"\nFound existing results at: {output_json_path}")
        user_input = input("Do you want to (L)oad existing results or (R)egenerate them? [L/R]: ").strip().upper()

        if user_input == 'L':
            print("\nLoading existing results from JSON...")
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} result entries")

            # Skip directly to visualization
            print("\n" + "="*100)
            print("Calculating overall statistics...")
            print("="*100)
            overall_stats = calculate_overall_stats(all_results)

            # Print results
            print("\nOverall Statistics:")
            print("-"*100)
            print(f"{'Model':<30} {'Move Acc':<12} {'Puzzle Acc':<12} {'Avg Time':<12}")
            print("-"*100)
            sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['avg_time_per_move'])
            for model_name, stats in sorted_models:
                print(f"{model_name:<30} {stats['move_accuracy']:>10.1f}%  {stats['puzzle_accuracy']:>10.1f}%  {stats['avg_time_per_move']:>10.4f}s")

            # Generate graphics
            print("\n" + "="*100)
            print("Generating bar charts...")
            print("="*100)

            graphs_dir = os.path.join(script_dir, "graphs")
            os.makedirs(graphs_dir, exist_ok=True)

            generate_response_time_bar_chart(overall_stats, graphs_dir)
            generate_move_accuracy_bar_chart(overall_stats, graphs_dir)
            generate_puzzle_accuracy_bar_chart(overall_stats, graphs_dir)
            generate_output_tokens_bar_chart(all_results, graphs_dir)

            print("\n" + "="*100)
            print("Chart generation complete!")
            print("All charts saved successfully!")
            print("="*100)
            return

    # Load puzzles and existing OpenAI results
    print("\nLoading puzzles and OpenAI results from openai_models_speeds.json...")
    puzzles_by_type = load_puzzles_from_speeds_json(speeds_file)
    print(f"Loaded {len(puzzles_by_type)} puzzle types")

    # Load existing OpenAI model results
    with open(speeds_file, 'r', encoding='utf-8') as f:
        openai_speeds_data = json.load(f)

    all_results = {}

    # Add OpenAI results to all_results
    print("\nAdding existing OpenAI model results...")
    for puzzle_type, models_data in openai_speeds_data.items():
        for model_key, model_data in models_data.items():
            model_name = model_data['model']
            reasoning = model_data.get('reasoning_effort', None)

            # Create display name with reasoning effort
            if reasoning:
                display_name = f"{model_name} ({reasoning})"
            else:
                display_name = model_name

            # Update the model field to include reasoning effort for consistency
            model_data_copy = deepcopy(model_data)
            model_data_copy['model'] = display_name

            all_results[f"{puzzle_type}_{model_key}"] = model_data_copy
            print(f"  Added {display_name} for {puzzle_type}")

    # Test Stockfish 17
    print("\n" + "="*100)
    print("Testing Stockfish 17 (think_time=1.0s)...")
    print("="*100)

    stockfish_tester = StockfishTester(stockfish_path, think_time=1.0)
    stockfish_tester.start_engine()

    for puzzle_type, puzzles in puzzles_by_type.items():
        print(f"\nTesting on {puzzle_type}...")
        result = test_model_on_puzzles(stockfish_tester, puzzles, "Stockfish 17 (t=1.0s)")
        all_results[f"{puzzle_type}_stockfish"] = result

    # Keep Stockfish engine running for alternative checking
    # Don't stop it yet

    # Test Caissa models
    for caissa_name, checkpoint_path in caissa_checkpoints.items():
        print("\n" + "="*100)
        print(f"Testing {caissa_name}...")
        print("="*100)

        caissa_tester = CaissaTester(checkpoint_path, caissa_name)
        caissa_tester.load_model()

        for puzzle_type, puzzles in puzzles_by_type.items():
            print(f"\nTesting on {puzzle_type}...")
            result = test_model_on_puzzles(caissa_tester, puzzles, caissa_name)
            all_results[f"{puzzle_type}_{caissa_name}"] = result

    # Check for alternative moves using Stockfish
    print("\n" + "="*100)
    print("Checking for alternative correct moves...")
    print("="*100)

    if stockfish_tester.engine:
        alt_checker = AlternativeMoveChecker(stockfish_tester.engine)
        all_results = check_alternatives_and_update_results(all_results, alt_checker)
    else:
        print("Warning: Stockfish engine not available for alternative checking")

    # Now stop Stockfish
    stockfish_tester.stop_engine()

    # Save results to JSON
    print("\n" + "="*100)
    print("Saving detailed results to JSON...")
    print("="*100)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {output_json_path}")

    # Print detailed results
    print_detailed_results(all_results)

    # Calculate overall stats (after alternative checking)
    print("\n" + "="*100)
    print("Calculating overall statistics (with alternatives)...")
    print("="*100)
    overall_stats = calculate_overall_stats(all_results)

    # Print results
    print("\nOverall Statistics (with alternative moves counted):")
    print("-"*100)
    print(f"{'Model':<30} {'Move Acc':<12} {'Puzzle Acc':<12} {'Avg Time':<12}")
    print("-"*100)
    sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['avg_time_per_move'])
    for model_name, stats in sorted_models:
        print(f"{model_name:<30} {stats['move_accuracy']:>10.1f}%  {stats['puzzle_accuracy']:>10.1f}%  {stats['avg_time_per_move']:>10.4f}s")

    # Generate graphics
    print("\n" + "="*100)
    print("Generating bar charts...")
    print("="*100)

    graphs_dir = os.path.join(script_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    generate_response_time_bar_chart(overall_stats, graphs_dir)
    generate_move_accuracy_bar_chart(overall_stats, graphs_dir)
    generate_puzzle_accuracy_bar_chart(overall_stats, graphs_dir)
    generate_output_tokens_bar_chart(all_results, graphs_dir)

    print("\n" + "="*100)
    print("Detailed speed comparison complete!")
    print("All charts saved successfully!")
    print(f"Results saved to: {output_json_path}")
    print("="*100)


if __name__ == "__main__":
    main()
