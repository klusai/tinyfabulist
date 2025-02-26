import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def plot_model_averages(args):
    """
    Reads a JSONL file with evaluation scores, computes average scores for each model,
    and plots a grouped bar chart with annotated grade values.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    """
    # Dictionary to accumulate scores and counts per model.
    score_totals = defaultdict(lambda: {"grammar": 0, "creativity": 0, "consistency": 0, "count": 0})
    
    # Read and process the JSONL file.
    with open(args.jsonl, "r") as file:
        for line in file:
            data = json.loads(line)
            model = data["model"]
            evaluation = data["evaluation"]
            
            # Accumulate scores for each metric.
            score_totals[model]["grammar"] += evaluation.get("grammar", 0)
            score_totals[model]["creativity"] += evaluation.get("creativity", 0)
            score_totals[model]["consistency"] += evaluation.get("consistency", 0)
            score_totals[model]["count"] += 1

    # Compute the averages for each model.
    averages = {}
    for model, scores in score_totals.items():
        count = scores["count"]
        averages[model] = {
            "grammar": scores["grammar"] / count,
            "creativity": scores["creativity"] / count,
            "consistency": scores["consistency"] / count,
        }

    print("Averages by model:")
    for model, metrics in averages.items():
        print(f"{model}: {metrics}")
    
    # Plotting the results.
    metrics = ["grammar", "creativity", "consistency"]
    models = list(averages.keys())
    x = np.arange(len(models))  # positions for each model on the x-axis
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each metric.
    colors = ['#1DB954', '#191414', '#535353']
        
    # Plot each metric as a group of bars.
    for i, metric in enumerate(metrics):
        # Get metric values for each model.
        values = [averages[model][metric] for model in models]
        # Calculate offset so that bars are centered for each model.
        offset = (i - 1) * width  
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i])
        
        # Annotate each bar with its average score.
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Average Evaluation Scores by Model", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 10)  # Adjust according to your scoring scale
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def add_comparator_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser('compare', help='Compare model results.')
    generate_parser.add_argument('--jsonl', default="evaluate.jsonl", help='Generate fable prompts')
    generate_parser.set_defaults(func=plot_model_averages)
