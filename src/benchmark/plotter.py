import matplotlib.pyplot as plt
import numpy as np
from src.benchmark.evaluation_metrics import EvaluationMetrics 

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


if __name__ == '__main__':
    # Example usage:
    # Create multiple EvaluationMetrics instances. They can use the same or different JSON files.
    metrics1 = EvaluationMetrics(
        evaluation_filename='evaluation_results.json', 
        diversity_filename='diversity_evaluation.json'
    )
    metrics2 = EvaluationMetrics(
        evaluation_filename='evaluation_results.json', 
        diversity_filename='diversity_evaluation.json'
    )

    evaluation_metrics_list = [metrics1, metrics2]
    labels = ["Model1", "Model2"]

    # Initialize the plotter with the list of evaluation metric objects and plot the grouped scores.
    plotter = Plotter(evaluation_metrics_list, labels)
    plotter.plot_grouped_by_model()
