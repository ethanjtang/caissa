import json
import os
from typing import Dict, List, Any
from collections import defaultdict

'''
Compare all chess models' (GPTs, Stockfish, Caissas) results across different puzzle themes.
Generates comprehensive statistics and comparison tables.
'''

def load_all_results(project_root: str) -> Dict[str, Any]:
    """Load all result JSON files."""
    results = {}

    # Load OpenAI models results
    openai_file = os.path.join(project_root, "openai_models_n=100_results.json")
    if os.path.exists(openai_file):
        with open(openai_file, 'r', encoding='utf-8') as f:
            results['openai'] = json.load(f)

    # Load Stockfish results
    stockfish_file = os.path.join(project_root, "stockfish17_results.json")
    if os.path.exists(stockfish_file):
        with open(stockfish_file, 'r', encoding='utf-8') as f:
            results['stockfish'] = json.load(f)

    # Load Caissa iteration results
    caissa_iterations = ['500k', '1m', '1.5m']
    for iteration in caissa_iterations:
        caissa_file = os.path.join(project_root, f"caissa-iters-{iteration}_results.json")
        if os.path.exists(caissa_file):
            with open(caissa_file, 'r', encoding='utf-8') as f:
                results[f'caissa-{iteration}'] = json.load(f)

    return results


def extract_model_data(all_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict]]:
    """Extract and organize model data by puzzle type and model.

    Returns:
        Dict mapping puzzle_type -> model_name -> summary_stats
    """
    organized = defaultdict(dict)

    # Process OpenAI models
    if 'openai' in all_results:
        for puzzle_type, models_data in all_results['openai'].items():
            for model_key, model_data in models_data.items():
                model_name = model_data['model']
                reasoning = model_data.get('reasoning_effort', None)

                # Create display name
                if reasoning:
                    display_name = f"{model_name} ({reasoning})"
                else:
                    display_name = model_name

                organized[puzzle_type][display_name] = {
                    'summary': model_data['summary'],
                    'type': 'openai',
                    'model_key': model_key
                }

    # Process Stockfish
    if 'stockfish' in all_results:
        for puzzle_type, sf_data in all_results['stockfish'].items():
            display_name = f"Stockfish 17 (t={sf_data['think_time']}s)"
            organized[puzzle_type][display_name] = {
                'summary': sf_data['summary'],
                'type': 'stockfish',
                'model_key': 'stockfish17'
            }

    # Process Caissa iterations
    caissa_keys = [key for key in all_results.keys() if key.startswith('caissa-')]
    for caissa_key in caissa_keys:
        for puzzle_type, caissa_data in all_results[caissa_key].items():
            # Extract the model name from the data (e.g., 'caissa-iters-500k')
            model_name = caissa_data.get('model', caissa_key)
            display_name = model_name

            organized[puzzle_type][display_name] = {
                'summary': caissa_data['summary'],
                'type': 'caissa',
                'model_key': caissa_key
            }

    return organized


def print_comparison_table(organized_data: Dict[str, Dict[str, Dict]]):
    """Print comprehensive comparison table across all models and puzzle types."""

    # Get all unique model names
    all_models = set()
    puzzle_types = list(organized_data.keys())

    for puzzle_type in puzzle_types:
        all_models.update(organized_data[puzzle_type].keys())

    all_models = sorted(list(all_models))

    print("\n" + "="*180)
    print("COMPREHENSIVE MODEL COMPARISON - ALL PUZZLE TYPES")
    print("="*180)

    # Header
    header = f"{'Puzzle Type':<20} {'Model':<15} {'Move Acc':<15} {'Puzzle Acc':<15} {'Sanity':<12} {'Avg Time/Move':<15} {'Valid Rate':<12} {'Details':<30}"
    print(header)
    print("-"*180)

    # Data rows
    for puzzle_type in sorted(puzzle_types):
        models_in_puzzle = organized_data[puzzle_type]

        for model_name in sorted(models_in_puzzle.keys()):
            model_data = models_in_puzzle[model_name]
            summary = model_data['summary']

            # Format statistics
            move_acc = f"{summary['move_accuracy']:.2f}%"
            move_detail = f"({summary['moves_correct']}/{summary['moves_total']})"
            move_acc_full = f"{move_acc} {move_detail}"

            puzzle_acc = f"{summary['puzzle_accuracy']:.2f}%"
            puzzle_detail = f"({summary['puzzles_correct']}/{summary['puzzles_total']})"
            puzzle_acc_full = f"{puzzle_acc} {puzzle_detail}"

            # Sanity rate = valid moves / total moves
            sanity_rate = f"{summary['move_valid_rate']:.2f}%"

            avg_time = f"{summary['avg_response_time_per_move']:.4f}s"
            valid_rate = f"{summary['move_valid_rate']:.2f}%"

            # Additional details based on model type
            if model_data['type'] == 'openai':
                total_tokens = summary.get('total_tokens', 0)
                details = f"Tokens: {total_tokens:,}"
            elif model_data['type'] == 'stockfish':
                details = f"Think time: 1.0s"
            else:
                details = f"GPU inference"

            row = f"{puzzle_type:<20} {model_name:<15} {move_acc_full:<15} {puzzle_acc_full:<15} {sanity_rate:<12} {avg_time:<15} {valid_rate:<12} {details:<30}"
            print(row)

        print("-"*180)

    print("="*180)


def calculate_overall_stats(organized_data: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Calculate overall statistics across all puzzle types for each model."""

    model_totals = defaultdict(lambda: {
        'total_moves': 0,
        'total_moves_correct': 0,
        'total_moves_valid': 0,
        'total_puzzles': 0,
        'total_puzzles_correct': 0,
        'total_response_time': 0.0,
        'total_tokens': 0,
        'puzzle_types_tested': 0
    })

    # Aggregate across all puzzle types
    for puzzle_type, models_data in organized_data.items():
        for model_name, model_data in models_data.items():
            summary = model_data['summary']

            totals = model_totals[model_name]
            totals['total_moves'] += summary['moves_total']
            totals['total_moves_correct'] += summary['moves_correct']
            totals['total_moves_valid'] += summary['moves_valid']
            totals['total_puzzles'] += summary['puzzles_total']
            totals['total_puzzles_correct'] += summary['puzzles_correct']
            totals['total_response_time'] += summary['total_response_time']
            totals['total_tokens'] += summary.get('total_tokens', 0)
            totals['puzzle_types_tested'] += 1

    # Calculate percentages
    overall_stats = {}
    for model_name, totals in model_totals.items():
        # Sanity rate = valid moves / total moves
        sanity_rate = 100 * totals['total_moves_valid'] / totals['total_moves'] if totals['total_moves'] > 0 else 0

        overall_stats[model_name] = {
            'total_moves': totals['total_moves'],
            'move_accuracy': 100 * totals['total_moves_correct'] / totals['total_moves'] if totals['total_moves'] > 0 else 0,
            'move_valid_rate': sanity_rate,
            'sanity_rate': sanity_rate,
            'total_puzzles': totals['total_puzzles'],
            'puzzle_accuracy': 100 * totals['total_puzzles_correct'] / totals['total_puzzles'] if totals['total_puzzles'] > 0 else 0,
            'avg_time_per_move': totals['total_response_time'] / totals['total_moves'] if totals['total_moves'] > 0 else 0,
            'total_time': totals['total_response_time'],
            'total_tokens': totals['total_tokens'],
            'puzzle_types_tested': totals['puzzle_types_tested']
        }

    return overall_stats


def print_overall_summary(overall_stats: Dict[str, Dict]):
    """Print overall summary across all puzzle types."""

    print("\n" + "="*160)
    print("OVERALL PERFORMANCE SUMMARY - AGGREGATED ACROSS ALL PUZZLE TYPES")
    print("="*160)

    header = f"{'Model':<35} {'Move Acc':<15} {'Puzzle Acc':<15} {'Sanity':<12} {'Avg Time/Move':<15} {'Total Time':<12} {'Puzzles':<10}"
    print(header)
    print("-"*160)

    # Sort by puzzle accuracy (descending)
    sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['puzzle_accuracy'], reverse=True)

    for model_name, stats in sorted_models:
        move_acc = f"{stats['move_accuracy']:.2f}%"
        puzzle_acc = f"{stats['puzzle_accuracy']:.2f}%"
        sanity = f"{stats['sanity_rate']:.2f}%"
        avg_time = f"{stats['avg_time_per_move']:.4f}s"
        total_time = f"{stats['total_time']:.2f}s"
        # tokens = f"{stats['total_tokens']:,}" if stats['total_tokens'] > 0 else "N/A"
        puzzles = f"{stats['total_puzzles']}"

        row = f"{model_name:<35} {move_acc:<15} {puzzle_acc:<15} {sanity:<12} {avg_time:<15} {total_time:<12} {puzzles:<10}"
        print(row)

    print("="*160)


def print_puzzle_type_rankings(organized_data: Dict[str, Dict[str, Dict]]):
    """Print rankings for each puzzle type."""

    print("\n" + "="*120)
    print("RANKINGS BY PUZZLE TYPE (Sorted by Puzzle Accuracy)")
    print("="*120)

    for puzzle_type in sorted(organized_data.keys()):
        print(f"\n{puzzle_type.upper().replace('_', ' ')}:")
        print("-"*120)

        models_data = organized_data[puzzle_type]

        # Sort by puzzle accuracy
        sorted_models = sorted(models_data.items(), key=lambda x: x[1]['summary']['puzzle_accuracy'], reverse=True)

        print(f"{'Rank':<6} {'Model':<35} {'Puzzle Acc':<15} {'Move Acc':<15} {'Time/Move':<15}")
        print("-"*120)

        for rank, (model_name, model_data) in enumerate(sorted_models, 1):
            summary = model_data['summary']
            puzzle_acc = f"{summary['puzzle_accuracy']:.2f}%"
            move_acc = f"{summary['move_accuracy']:.2f}%"
            avg_time = f"{summary['avg_response_time_per_move']:.4f}s"

            print(f"{rank:<6} {model_name:<35} {puzzle_acc:<15} {move_acc:<15} {avg_time:<15}")


def print_speed_analysis(overall_stats: Dict[str, Dict]):
    """Print speed analysis."""

    print("\n" + "="*100)
    print("SPEED ANALYSIS - Average Response Time per Move")
    print("="*100)

    # Sort by speed (ascending)
    sorted_by_speed = sorted(overall_stats.items(), key=lambda x: x[1]['avg_time_per_move'])

    print(f"{'Rank':<6} {'Model':<25} {'Avg Time/Move':<20} {'Total Moves':<15} {'Total Time':<15}")
    print("-"*100)

    for rank, (model_name, stats) in enumerate(sorted_by_speed, 1):
        avg_time = f"{stats['avg_time_per_move']:.4f}s"
        total_moves = f"{stats['total_moves']}"
        total_time = f"{stats['total_time']:.2f}s"

        print(f"{rank:<6} {model_name:<25} {avg_time:<20} {total_moves:<15} {total_time:<15}")

    print("="*100)


def print_token_efficiency(overall_stats: Dict[str, Dict]):
    """Print token efficiency analysis for models that use tokens."""

    print("\n" + "="*100)
    print("TOKEN EFFICIENCY - For Language Models")
    print("="*100)

    # Filter models with token usage
    models_with_tokens = {name: stats for name, stats in overall_stats.items() if stats['total_tokens'] > 0}

    if not models_with_tokens:
        print("No token-based models found.")
        return

    # Calculate tokens per move
    for name, stats in models_with_tokens.items():
        stats['tokens_per_move'] = stats['total_tokens'] / stats['total_moves'] if stats['total_moves'] > 0 else 0

    # Sort by tokens per move
    sorted_by_tokens = sorted(models_with_tokens.items(), key=lambda x: x[1]['tokens_per_move'])

    print(f"{'Model':<35} {'Total Tokens':<15} {'Tokens/Move':<15} {'Accuracy':<15} {'Moves':<10}")
    print("-"*100)

    for model_name, stats in sorted_by_tokens:
        total_tokens = f"{stats['total_tokens']:,}"
        tokens_per_move = f"{stats['tokens_per_move']:.1f}"
        accuracy = f"{stats['move_accuracy']:.2f}%"
        moves = f"{stats['total_moves']}"

        print(f"{model_name:<35} {total_tokens:<15} {tokens_per_move:<15} {accuracy:<15} {moves:<10}")

    print("="*100)


def print_sanity_analysis(overall_stats: Dict[str, Dict], organized_data: Dict[str, Dict[str, Dict]]):
    """Print sanity check analysis - percentage of valid moves that don't result in illegal positions."""

    print("\n" + "="*120)
    print("SANITY ANALYSIS - Overall Sanity Rate by Model")
    print("="*120)
    print("Sanity Rate: Percentage of valid moves (valid moves / total moves)")
    print("-"*120)

    # Sort by sanity rate (descending)
    sorted_by_sanity = sorted(overall_stats.items(), key=lambda x: x[1]['sanity_rate'], reverse=True)

    header = f"{'Rank':<6} {'Model':<35} {'Sanity Rate':<15} {'Move Acc':<15} {'Puzzle Acc':<15} {'Total Moves':<12}"
    print(header)
    print("-"*120)

    for rank, (model_name, stats) in enumerate(sorted_by_sanity, 1):
        sanity = f"{stats['sanity_rate']:.2f}%"
        move_acc = f"{stats['move_accuracy']:.2f}%"
        puzzle_acc = f"{stats['puzzle_accuracy']:.2f}%"
        total_moves = f"{stats['total_moves']}"

        print(f"{rank:<6} {model_name:<35} {sanity:<15} {move_acc:<15} {puzzle_acc:<15} {total_moves:<12}")

    print("="*120)

    # Print sanity rate by puzzle type
    print("\n" + "="*140)
    print("SANITY ANALYSIS - Breakdown by Puzzle Type")
    print("="*140)

    for puzzle_type in sorted(organized_data.keys()):
        print(f"\n{puzzle_type.upper().replace('_', ' ')}:")
        print("-"*140)

        models_data = organized_data[puzzle_type]

        # Sort by sanity rate (move_valid_rate)
        sorted_models = sorted(models_data.items(),
                              key=lambda x: x[1]['summary']['move_valid_rate'],
                              reverse=True)

        print(f"{'Rank':<6} {'Model':<35} {'Sanity':<12} {'Move Acc':<15} {'Puzzle Acc':<15}")
        print("-"*140)

        for rank, (model_name, model_data) in enumerate(sorted_models, 1):
            summary = model_data['summary']
            sanity = f"{summary['move_valid_rate']:.2f}%"
            move_acc = f"{summary['move_accuracy']:.2f}%"
            puzzle_acc = f"{summary['puzzle_accuracy']:.2f}%"

            print(f"{rank:<6} {model_name:<35} {sanity:<12} {move_acc:<15} {puzzle_acc:<15}")

    print("="*140)


def main():
    """Generate comprehensive comparison report."""

    # Script is in the main directory with JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    print("="*160)
    print("CHESS MODEL COMPARISON REPORT")
    print("Comparing: OpenAI GPT Models, Stockfish 17, and Caissa")
    print("="*160)

    # Load all results
    print("\nLoading results...")
    all_results = load_all_results(project_root)

    if not all_results:
        print("Error: No result files found!")
        return

    print(f"Loaded results from:")
    for source in all_results.keys():
        print(f"  - {source}")

    # Extract and organize data
    organized_data = extract_model_data(all_results)

    # Calculate overall statistics
    overall_stats = calculate_overall_stats(organized_data)

    # Print all analyses
    print_comparison_table(organized_data)
    print_overall_summary(overall_stats)
    print_puzzle_type_rankings(organized_data)
    print_speed_analysis(overall_stats)
    print_token_efficiency(overall_stats)
    print_sanity_analysis(overall_stats, organized_data)

    # Save text report
    print("\n" + "="*160)
    print("Report generation complete!")
    print("="*160)


if __name__ == "__main__":
    main()
