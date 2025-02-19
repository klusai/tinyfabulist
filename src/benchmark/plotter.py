import matplotlib.pyplot as plt
import numpy as np
from src.benchmark.evaluation_metrics import EvaluationMetrics 
import os
import glob

class Plotter:
    def __init__(self, evaluation_metrics_list, labels=None):
        """
        Initializes the Plotter for multiple EvaluationMetrics objects.

        Args:
            evaluation_metrics_list (list): List of EvaluationMetrics instances.
            labels (list, optional): List of labels corresponding to each EvaluationMetrics object.
                                     If not provided, default labels ("Object 1", "Object 2", etc.) are used.
        """
        self.evaluation_metrics_list = evaluation_metrics_list
        if labels is None:
            self.labels = [f"Object {i+1}" for i in range(len(evaluation_metrics_list))]
        else:
            self.labels = labels

    def plot_grouped_by_model(self):
        """
        Plots a grouped bar chart where each group represents one EvaluationMetrics object,
        and within each group the scores for Grammar, Creativity, Consistency, and Diversity are shown.
        """
        # Define the field names (metrics)
        fields = ["Grammar", "Creativity", "Consistency", "Diversity"]

        # Prepare the data for each evaluation model:
        # Each entry in 'data' is a list of scores for [grammar, creativity, consistency, diversity]
        data = []
        for metrics_obj in self.evaluation_metrics_list:
            data.append([
                metrics_obj.grammar,
                metrics_obj.creativity,
                metrics_obj.consistency,
                metrics_obj.diversity_score
            ])

        n_objects = len(self.evaluation_metrics_list)
        n_fields = len(fields)

        # Create x positions for the groups (each group corresponds to a model)
        x = np.arange(n_objects)
        # Define the width of each bar (so that all fields fit within one group)
        width = 0.8 / n_fields

        fig, ax = plt.subplots(figsize=(10, 6))

        # For each field, plot its bars across all evaluation models.
        for i, field in enumerate(fields):
            # Calculate positions for each field's bar within a group.
            # This centers the group of bars around each x position.
            offsets = x - 0.4 + width/2 + i * width
            # Extract the scores for the current field from all models.
            field_scores = [entry[i] for entry in data]
            bars = ax.bar(offsets, field_scores, width, label=field)
            # Annotate each bar with its value.
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # Offset text by 3 points above the bar.
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels)
        ax.set_ylabel("Score (1 to 10)")
        ax.set_title("Evaluation Metrics by Model")
        ax.set_ylim(0, 10)
        ax.legend()

        plt.show()


def plot_from_artifacts(artifacts_dir):
    """
    Finds JSON files in the artifacts directory, groups those with both
    'diversity_evaluation.json' and 'evaluation_results.json' having the same prefix,
    and plots the evaluation metrics for each group.

    Args:
        artifacts_dir (str): Path to the artifacts directory.
    """

    # Find all JSON files in the artifacts directory
    json_files = glob.glob(os.path.join(artifacts_dir, "*.json"))

    diversity_results = {}
    evaluation_results = {}

    for json_file in json_files:
        if json_file.endswith("evaluation_results.json"):
            prefix = json_file.replace("evaluation_results.json", "")
            evaluation_results[prefix] = json_file
        elif json_file.endswith("diversity_evaluation.json"):
            prefix = json_file.replace("diversity_evaluation.json", "")
            diversity_results[prefix] = json_file

    # Group evaluation and diversity results by common prefix
    grouped_results = []
    for prefix, eval_file in evaluation_results.items():
        if prefix in diversity_results:
            div_file = diversity_results[prefix]
            grouped_results.append((prefix, eval_file, div_file))

    # Create EvaluationMetrics objects and labels for plotting
    metrics_list = []
    labels = []
    for prefix, eval_file, div_file in grouped_results:
        metrics = EvaluationMetrics(div_file, eval_file)
        metrics_list.append(metrics)
        prefix = prefix.split("/")[-1]
        prefix = " ".join(prefix.split("-")[:-1])
        labels.append(prefix)

    # Plot the grouped results
    if metrics_list:
        plotter = Plotter(metrics_list, labels=labels)
        plotter.plot_grouped_by_model()
    else:
        print("No matching evaluation and diversity results found.")

if __name__ == '__main__':
    artifacts_directory = 'src/artifacts' 
    plot_from_artifacts(artifacts_directory)
