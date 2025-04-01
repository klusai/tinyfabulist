import argparse
import json
import os
import time
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Logger Setup ---
logger = logging.getLogger("TinyFabulist")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- File Processing Module ---
class FileProcessor:
    @staticmethod
    def identify_files(input_path: str):
        """
        Identify evaluation and fable files in a given file or directory.
        Returns: tuple (standard_eval_files, translation_eval_files, tf_files)
        """
        standard_eval_files = []
        translation_eval_files = []
        tf_files = []

        def process_file(file_path: str):
            base_name = os.path.basename(file_path)
            if "eval_e" in base_name:
                is_translation = False
                try:
                    with open(file_path, "r") as f:
                        for i, line in enumerate(f):
                            if i > 10:
                                break
                            try:
                                data = json.loads(line)
                                if "evaluation" in data and "translation_accuracy" in data["evaluation"]:
                                    is_translation = True
                                    break
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                if is_translation:
                    translation_eval_files.append(file_path)
                else:
                    standard_eval_files.append(file_path)
            elif base_name.startswith("tf_fables"):
                tf_files.append(file_path)

        if os.path.isfile(input_path):
            process_file(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".jsonl"):
                        process_file(os.path.join(root, file))
        else:
            logger.warning("Provided input path does not exist.")

        return standard_eval_files, translation_eval_files, tf_files

    @staticmethod
    def parse_fables_data(tf_files: list) -> dict:
        """
        Parse data from tf_fables files and return aggregated model stats.
        """
        tf_model_stats = defaultdict(lambda: {
            "count": 0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_inference_time": 0,
            "host_providers": defaultdict(int),
            "host_gpus": defaultdict(int),
            "host_gpu_vram": defaultdict(int),
            "host_dc_providers": defaultdict(int),
            "host_dc_locations": defaultdict(int),
            "host_cost_per_hour": 0.0,
        })

        for file_path in tf_files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        model = data.get("llm_name", "unknown").split("/")[-1]
                        stats = tf_model_stats[model]

                        stats["count"] += 1
                        stats["avg_input_tokens"] += data.get("llm_input_tokens", 0)
                        stats["avg_output_tokens"] += data.get("llm_output_tokens", 0)
                        stats["avg_inference_time"] += data.get("llm_inference_time", 0)

                        stats["host_providers"][data.get("host_provider", "unknown")] += 1
                        stats["host_gpus"][data.get("host_gpu", "unknown")] += 1
                        stats["host_dc_providers"][data.get("host_dc_provider", "unknown")] += 1
                        stats["host_dc_locations"][data.get("host_dc_location", "unknown")] += 1
                        stats["host_cost_per_hour"] += data.get("host_cost_per_hour", 0.0)
                        stats["host_gpu_vram"][data.get("host_gpu_vram", 0)] += 1
            except Exception as e:
                logger.error(f"Error parsing tf file {file_path}: {e}")

        # Calculate averages
        for model, stats in tf_model_stats.items():
            count = stats["count"]
            if count:
                stats["avg_input_tokens"] /= count
                stats["avg_output_tokens"] /= count
                stats["avg_inference_time"] /= count
                stats["host_cost_per_hour"] /= count

        return tf_model_stats


# --- Evaluation Module ---
class Evaluator:
    @staticmethod
    def parse_evaluation_data(eval_files: list) -> dict:
        """
        Parse standard evaluation file(s) into aggregated scores.
        """
        score_totals = defaultdict(lambda: {
            "grammar": 0,
            "creativity": 0,
            "moral_clarity": 0,
            "adherence_to_prompt": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "count": 0,
        })

        for file_path in eval_files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        model = data.get("llm_name", "unknown").split("/")[-1]
                        if "evaluation" in data:
                            eval_data = data["evaluation"]
                            score_totals[model]["grammar"] += eval_data.get("grammar", 0)
                            score_totals[model]["creativity"] += eval_data.get("creativity", 0)
                            score_totals[model]["moral_clarity"] += eval_data.get("moral_clarity", 0)
                            score_totals[model]["adherence_to_prompt"] += eval_data.get("adherence_to_prompt", 0)

                        score_totals[model]["input_tokens"] += data.get("llm_input_tokens", 0)
                        score_totals[model]["output_tokens"] += data.get("llm_output_tokens", 0)
                        score_totals[model]["inference_time"] += data.get("llm_inference_time", 0)
                        score_totals[model]["count"] += 1
            except Exception as e:
                logger.error(f"Error parsing evaluation file {file_path}: {e}")
        return score_totals

    @staticmethod
    def parse_translation_evaluation_data(translation_eval_files: list) -> dict:
        """
        Parse translation evaluation file(s) into aggregated scores.
        """
        translation_score_totals = defaultdict(lambda: {
            "translation_accuracy": 0,
            "fluency": 0,
            "style_preservation": 0,
            "moral_clarity": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "count": 0,
        })

        for file_path in translation_eval_files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        model = data.get("llm_name", "unknown").split("/")[-1]
                        if "evaluation" in data:
                            eval_data = data["evaluation"]
                            if "translation_accuracy" in eval_data:
                                translation_score_totals[model]["translation_accuracy"] += eval_data.get("translation_accuracy", 0)
                                translation_score_totals[model]["fluency"] += eval_data.get("fluency", 0)
                                translation_score_totals[model]["style_preservation"] += eval_data.get("style_preservation", 0)
                                translation_score_totals[model]["moral_clarity"] += eval_data.get("moral_clarity", 0)
                                translation_score_totals[model]["input_tokens"] += data.get("llm_input_tokens", 0)
                                translation_score_totals[model]["output_tokens"] += data.get("llm_output_tokens", 0)
                                translation_score_totals[model]["inference_time"] += data.get("llm_inference_time", 0)
                                translation_score_totals[model]["count"] += 1
            except Exception as e:
                logger.error(f"Error parsing translation evaluation file {file_path}: {e}")
        return translation_score_totals

    @staticmethod
    def compute_averages(score_totals: dict) -> dict:
        """
        Compute average scores from aggregated evaluation totals.
        """
        averages = {}
        for model, scores in score_totals.items():
            count = scores["count"]
            if count:
                averages[model] = {
                    "grammar": scores["grammar"] / count,
                    "creativity": scores["creativity"] / count,
                    "moral_clarity": scores["moral_clarity"] / count,
                    "adherence_to_prompt": scores["adherence_to_prompt"] / count,
                    "average_score_(mean)": (scores["grammar"] + scores["creativity"] +
                                             scores["moral_clarity"] + scores["adherence_to_prompt"]) / (4 * count),
                    "input_tokens": scores["input_tokens"] / count,
                    "output_tokens": scores["output_tokens"] / count,
                    "inference_time": scores["inference_time"] / count,
                    "count": count,
                }
        return averages

    @staticmethod
    def compute_translation_averages(translation_score_totals: dict) -> dict:
        """
        Compute average scores for translation evaluations.
        """
        translation_averages = {}
        for model, scores in translation_score_totals.items():
            count = scores["count"]
            if count:
                ta = scores["translation_accuracy"] / count
                fluency = scores["fluency"] / count
                style = scores["style_preservation"] / count
                moral = scores["moral_clarity"] / count
                translation_averages[model] = {
                    "translation_accuracy": ta,
                    "fluency": fluency,
                    "style_preservation": style,
                    "moral_clarity": moral,
                    "average_score_(mean)": (ta + fluency + style + moral) / 4,
                    "input_tokens": scores["input_tokens"] / count,
                    "output_tokens": scores["output_tokens"] / count,
                    "inference_time": scores["inference_time"] / count,
                    "count": count,
                }
        return translation_averages

    @staticmethod
    def get_max_key(counter_dict: dict) -> str:
        """Return the key with the maximum value from a dictionary (or 'unknown' if empty)."""
        return max(counter_dict.items(), key=lambda x: x[1])[0] if counter_dict else "unknown"

    @staticmethod
    def create_fables_markdown_table(tf_model_stats: dict) -> str:
        """
        Create a markdown table summarizing tf_fables data.
        """
        header = [
            "# Consolidated Fable Statistics",
            "## Summary (All Files)",
            "| Model | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) | Primary Host Provider | Primary GPU | GPU VRAM (GB) | DC Provider | DC Location | Avg Cost per Hour ($) |",
            "|-------|-------|-----------------|------------------|---------------------|-----------------------|-------------|---------------|-------------|-------------|-----------------------|",
        ]
        rows = []
        for model, stats in sorted(tf_model_stats.items()):
            primary_provider = Evaluator.get_max_key(stats["host_providers"])
            primary_gpu = Evaluator.get_max_key(stats["host_gpus"])
            primary_gpu_vram = Evaluator.get_max_key(stats["host_gpu_vram"])
            primary_dc_provider = Evaluator.get_max_key(stats["host_dc_providers"])
            primary_dc_location = Evaluator.get_max_key(stats["host_dc_locations"])
            row = (f"| {model} | {stats['count']} | {stats['avg_input_tokens']:.1f} | "
                   f"{stats['avg_output_tokens']:.1f} | {stats['avg_inference_time']:.2f} | "
                   f"{primary_provider} | {primary_gpu} | {primary_gpu_vram} | "
                   f"{primary_dc_provider} | {primary_dc_location} | {stats['host_cost_per_hour']:.2f} |")
            rows.append(row)
        return "\n".join(header + [""] + rows)

    @staticmethod
    def create_evaluation_markdown_table(averages: dict) -> str:
        """
        Create a markdown table from evaluation averages.
        """
        header = [
            "## Evaluation Averages",
            "",
            "| Model | Grammar | Creativity | Moral Clarity | Adherence to Prompt | Average Score (Mean) | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) |",
            "|-------|---------|------------|---------------|---------------------|----------------------|-------|------------------|-------------------|------------------------|",
        ]
        rows = []
        for model, metrics in averages.items():
            row = (f"| {model} | {metrics['grammar']:.2f} | {metrics['creativity']:.2f} | "
                   f"{metrics['moral_clarity']:.2f} | {metrics['adherence_to_prompt']:.2f} | "
                   f"{metrics['average_score_(mean)']:.2f} | {metrics['count']} | "
                   f"{metrics['input_tokens']:.1f} | {metrics['output_tokens']:.1f} | "
                   f"{metrics['inference_time']:.2f} |")
            rows.append(row)
        return "\n".join(header + [""] + rows)

    @staticmethod
    def create_translation_evaluation_markdown_table(translation_averages: dict) -> str:
        """
        Create a markdown table from translation evaluation averages.
        """
        header = [
            "## Translation Evaluation Averages",
            "",
            "| Model | Translation Accuracy | Fluency | Style Preservation | Moral Clarity | Average Score (Mean) | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) |",
            "|-------|----------------------|---------|--------------------|---------------|----------------------|-------|------------------|-------------------|------------------------|",
        ]
        rows = []
        for model, metrics in translation_averages.items():
            row = (f"| {model} | {metrics['translation_accuracy']:.2f} | {metrics['fluency']:.2f} | "
                   f"{metrics['style_preservation']:.2f} | {metrics['moral_clarity']:.2f} | "
                   f"{metrics['average_score_(mean)']:.2f} | {metrics['count']} | "
                   f"{metrics['input_tokens']:.1f} | {metrics['output_tokens']:.1f} | "
                   f"{metrics['inference_time']:.2f} |")
            rows.append(row)
        return "\n".join(header + [""] + rows)


# --- Visualization Module ---
class Visualizer:
    # ----- Plotly Visualizations -----
    @staticmethod
    def generate_plotly_visualizations(averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str):
        models_list = list(averages.keys())
        metrics_list = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt", "average_score_(mean)"]
        fig = Visualizer.create_plotly_figure(averages, tf_model_stats, metrics_list, models_list)

        # Save outputs if required
        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_eval_plot_{timestamp}.png")
            html_filename = os.path.join(stats_folder, f"tf_stats_eval_plot_{timestamp}.html")
            fig.write_image(plot_filename)
            fig.write_html(html_filename)
            logger.info(f"Evaluation plots saved to {plot_filename} and {html_filename}")
        if output_mode in ["terminal", "both"]:
            fig.show()

    @staticmethod
    def create_plotly_figure(averages: dict, tf_model_stats: dict, metrics_list: list, models_list: list) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.2,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        # Define colors for evaluation metrics
        eval_colors = {
            "grammar": "#1DB954",
            "creativity": "#191414",
            "moral_clarity": "#535353",
            "adherence_to_prompt": "#B3B3B3",
            "average_score_(mean)": "#E60012",
        }
        for i, metric in enumerate(metrics_list):
            values = [averages[model][metric] for model in models_list]
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=values,
                    name=metric.replace("_", " ").title(),
                    text=[f"{val:.2f}" for val in values],
                    textposition="auto",
                    marker_color=eval_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                    hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',  
                ),
                row=1, col=1
            )
        # Performance metrics: prefer tf_model_stats if available
        if tf_model_stats:
            performance_metrics = ["avg_input_tokens", "avg_output_tokens", "avg_inference_time"]
            source = tf_model_stats
        else:
            performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
            source = averages
        perf_colors = {
            "input_tokens": "#4CAF50",
            "output_tokens": "#2196F3",
            "inference_time": "#FFC107",
        }
        for i, metric in enumerate(performance_metrics):
            values = [source.get(model, {}).get(metric, 0) for model in models_list]
            display_name = metric.replace("avg_", "").replace("_", " ").title()
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=values,
                    name=display_name,
                    text=[f"{val:.1f}" for val in values],
                    textposition="auto",
                    marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                    hovertemplate=f"%{{x}}<br>{display_name}: %{{y:.1f}}<extra></extra>",
                ),
                row=2, col=1
            )
        Visualizer.configure_plotly_layout(fig)
        return fig

    @staticmethod
    def configure_plotly_layout(fig: go.Figure):
        fig.update_layout(
            title_text="Model Evaluation and Performance Analytics",
            title_x=0.5,  # Center the title
            title_font=dict(size=24),  # Increase title font size
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),  # Increase legend font size
            ),
            template="plotly_white",
            height=1080,  # Full height
            width=1920,   # Full width
            margin=dict(l=80, r=80, t=120, b=80),  # Padding around the plot
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,  # Increase hover label font size
                font_family="Arial",
            ),
        )
        # For the evaluation scores subplot (row 1) with fixed range [0,10]
        fig.update_yaxes(
            row=1, col=1,
            title_text="Score",
            range=[0, 10],
            title_font=dict(size=16),
            tickfont=dict(size=14),
            automargin=True,
            title_standoff=40,  # Increase space between y-axis title and tick labels
        )
        # For the performance metrics subplot (row 2), let Plotly auto-scale
        fig.update_yaxes(
            row=2, col=1,
            title_text="Value",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            automargin=True,
            title_standoff=40,  # Increase space between y-axis title and tick labels
        )


    @staticmethod
    def generate_translation_plotly_visualizations(translation_averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str):
        models_list = list(translation_averages.keys())
        models_list_names = [model.split("_")[-1] for model in models_list]
        metrics_list = ["translation_accuracy", "fluency", "style_preservation", "moral_clarity", "average_score_(mean)"]

        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.2,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        trans_colors = {
            "translation_accuracy": "#1DB954",
            "fluency": "#191414",
            "style_preservation": "#535353",
            "moral_clarity": "#B3B3B3",
            "average_score_(mean)": "#E60012",
        }
        for i, metric in enumerate(metrics_list):
            values = [translation_averages[model][metric] for model in models_list]
            fig.add_trace(
                go.Bar(
                    x=models_list_names,
                    y=values,
                    name=metric.replace("_", " ").title(),
                    text=[f"{val:.2f}" for val in values],
                    textposition="auto",
                    marker_color=trans_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                    hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
                ),
                row=1, col=1
            )
        # Performance metrics
        if tf_model_stats:
            performance_metrics = ["avg_input_tokens", "avg_output_tokens", "avg_inference_time"]
            source = tf_model_stats
        else:
            performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
            source = translation_averages
        perf_colors = {
            "input_tokens": "#4CAF50",
            "output_tokens": "#2196F3",
            "inference_time": "#FFC107",
        }
        for i, metric in enumerate(performance_metrics):
            values = [source.get(model, {}).get(metric, 0) for model in models_list]
            display_name = metric.replace("avg_", "").replace("_", " ").title()
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=values,
                    name=display_name,
                    text=[f"{val:.1f}" for val in values],
                    textposition="auto",
                    marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                    hovertemplate=f"%{{x}}<br>{display_name}: %{{y:.1f}}<extra></extra>",
                ),
                row=2, col=1
            )
        fig.update_layout(
            title_text="Translation Evaluation and Performance Analytics",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            template="plotly_white",
            height=900,
            width=1000,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        )
        fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)

        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_translation_plot_{timestamp}.png")
            html_filename = os.path.join(stats_folder, f"tf_stats_translation_plot_{timestamp}.html")
            fig.write_image(plot_filename)
            fig.write_html(html_filename)
            logger.info(f"Translation evaluation plots saved to {plot_filename} and {html_filename}")
        if output_mode in ["terminal", "both"]:
            fig.show()

    # ----- Matplotlib Visualizations -----
    @staticmethod
    def generate_matplotlib_visualizations(averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str):
        models_list = list(averages.keys())
        metrics_list = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt", "average_score_(mean)"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))  # Increase figure size
        Visualizer.plot_evaluation_scores(ax1, averages, models_list, metrics_list)
        Visualizer.plot_performance_metrics(ax2, averages, tf_model_stats, models_list)
        plt.tight_layout()
        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_eval_plot_{timestamp}.png")
            plt.savefig(plot_filename)
            logger.info(f"Evaluation plot saved to {plot_filename}")
        if output_mode in ["terminal", "both"]:
            plt.show()

    @staticmethod
    def plot_evaluation_scores(ax, averages: dict, models_list: list, metrics_list: list):
        x = np.arange(len(models_list))
        width = 0.17
        colors = ["#1DB954", "#191414", "#535353", "#B3B3B3", "#E60012"]
        for i, metric in enumerate(metrics_list):
            values = [averages[model][metric] for model in models_list]
            offset = (i - 2) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(), color=colors[i % len(colors)])
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, fontsize=12)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    @staticmethod
    def plot_performance_metrics(ax, averages: dict, tf_model_stats: dict, models_list: list):
        if tf_model_stats:
            performance_metrics = ["avg_input_tokens", "avg_output_tokens", "avg_inference_time"]
            source = tf_model_stats
        else:
            performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
            source = averages
        x = np.arange(len(models_list))
        width_perf = 0.15
        colors_perf = ["#4CAF50", "#2196F3", "#FFC107"]
        for i, metric in enumerate(performance_metrics):
            values = [source.get(model, {}).get(metric, 0) for model in models_list]
            offset = (i - 1) * width_perf
            bars = ax.bar(x + offset, values, width_perf, label=metric.replace("avg_", "").replace("_", " ").title(), color=colors_perf[i % len(colors_perf)])
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")
        ax.set_title("Performance Metrics by Model", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    @staticmethod
    def generate_translation_matplotlib_visualizations(translation_averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str):
        models_list = list(translation_averages.keys())
        models_list_names = [model.split("_")[-1] for model in models_list]
        metrics_list = ["translation_accuracy", "fluency", "style_preservation", "moral_clarity", "average_score_(mean)"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))  # Increase figure size
        x = np.arange(len(models_list))
        width = 0.17
        colors = ["#1DB954", "#191414", "#535353", "#B3B3B3", "#E60012"]
        for i, metric in enumerate(metrics_list):
            values = [translation_averages[model][metric] for model in models_list]
            offset = (i - 2) * width
            bars = ax1.bar(x + offset, values, width, label=metric.replace("_", " ").title(), color=colors[i % len(colors)])
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, fontweight="bold")
        ax1.set_title("Average Translation Evaluation Scores by Model", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_list_names, fontsize=12)
        ax1.set_ylim(0, 10)
        ax1.legend(fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        if tf_model_stats:
            performance_metrics = ["avg_input_tokens", "avg_output_tokens", "avg_inference_time"]
            source = tf_model_stats
        else:
            performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
            source = translation_averages
        width_perf = 0.15
        colors_perf = ["#4CAF50", "#2196F3", "#FFC107"]
        for i, metric in enumerate(performance_metrics):
            values = [source.get(model, {}).get(metric, 0) for model in models_list]
            offset = (i - 1) * width_perf
            bars = ax2.bar(x + offset, values, width_perf, label=metric.replace("avg_", "").replace("_", " ").title(), color=colors_perf[i % len(colors_perf)])
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, fontweight="bold")
        ax2.set_title("Performance Metrics by Model", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models_list, fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_translation_plot_{timestamp}.png")
            plt.savefig(plot_filename)
            logger.info(f"Translation evaluation plot saved to {plot_filename}")
        if output_mode in ["terminal", "both"]:
            plt.show()


# --- Orchestrator Functions ---
def process_tf_files(tf_files, stats_folder, timestamp, output_mode) -> dict:
    tf_model_stats = FileProcessor.parse_fables_data(tf_files)
    md_table = Evaluator.create_fables_markdown_table(tf_model_stats)
    if output_mode in ["files", "both"]:
        md_tf_filename = os.path.join(stats_folder, f"tf_stats_fables_{timestamp}.md")
        with open(md_tf_filename, "w") as f:
            f.write(md_table)
        logger.info(f"Consolidated tf_fable report saved to {md_tf_filename}")
    if output_mode in ["terminal", "both"]:
        print("\n" + md_table + "\n")
    return tf_model_stats


def process_eval_files(eval_files, tf_model_stats, stats_folder, timestamp, output_mode, plot_mode="plotly"):
    score_totals = Evaluator.parse_evaluation_data(eval_files)
    averages = Evaluator.compute_averages(score_totals) 
    md_table = Evaluator.create_evaluation_markdown_table(averages)
    if output_mode in ["files", "both"]:
        md_eval_filename = os.path.join(stats_folder, f"tf_stats_eval_table_{timestamp}.md")
        with open(md_eval_filename, "w") as f:
            f.write(md_table)
        logger.info(f"Evaluation Markdown table saved to {md_eval_filename}")
    if output_mode in ["terminal", "both"]:
        print("\n" + md_table + "\n")
    if plot_mode.lower() == "plotly":
        Visualizer.generate_plotly_visualizations(averages, tf_model_stats, stats_folder, timestamp, output_mode)
    else:
        Visualizer.generate_matplotlib_visualizations(averages, tf_model_stats, stats_folder, timestamp, output_mode)


def process_translation_eval_files(translation_eval_files, tf_model_stats, stats_folder, timestamp, output_mode, plot_mode="plotly"):
    translation_score_totals = Evaluator.parse_translation_evaluation_data(translation_eval_files)
    translation_averages = Evaluator.compute_translation_averages(translation_score_totals)
    md_table = Evaluator.create_translation_evaluation_markdown_table(translation_averages)
    if output_mode in ["files", "both"]:
        md_trans_filename = os.path.join(stats_folder, f"tf_stats_translation_eval_table_{timestamp}.md")
        with open(md_trans_filename, "w") as f:
            f.write(md_table)
        logger.info(f"Translation evaluation table saved to {md_trans_filename}")
    if output_mode in ["terminal", "both"]:
        print("\n" + md_table + "\n")
    if plot_mode.lower() == "plotly":
        Visualizer.generate_translation_plotly_visualizations(translation_averages, tf_model_stats, stats_folder, timestamp, output_mode)
    else:
        Visualizer.generate_translation_matplotlib_visualizations(translation_averages, tf_model_stats, stats_folder, timestamp, output_mode)


def plot_model_averages(args):
    """
    Main orchestrator to process evaluation and tf_fables files.
    """
    standard_eval_files, translation_eval_files, tf_files = FileProcessor.identify_files(args.input)
    stats_folder = "data/stats/"
    if args.output_mode in ["files", "both"]:
        os.makedirs(stats_folder, exist_ok=True)
    timestamp = time.strftime("%y%m%d-%H%M%S")

    tf_model_stats = None
    if tf_files:
        tf_model_stats = process_tf_files(tf_files, stats_folder, timestamp, args.output_mode)
    if standard_eval_files:
        process_eval_files(standard_eval_files, tf_model_stats, stats_folder, timestamp, args.output_mode, args.plot_mode)
    else:
        logger.info("No standard evaluation files found.")
    if translation_eval_files:
        process_translation_eval_files(translation_eval_files, tf_model_stats, stats_folder, timestamp, args.output_mode, args.plot_mode)
    else:
        logger.info("No translation evaluation files found.")
    if not (standard_eval_files or translation_eval_files or tf_files):
        logger.info("No relevant files found.")


def add_stats_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        "stats",
        help="Compute aggregated statistics from evaluation JSONL files (file or directory)."
    )
    generate_parser.add_argument(
        "--input",
        default="evaluate.jsonl",
        help="Path to a JSONL file or a folder containing JSONL files"
    )
    generate_parser.add_argument(
        "--output-mode",
        choices=["terminal", "files", "both"],
        default="both",
        help="Control where to output results: terminal only, files only, or both (default: both)"
    )
    generate_parser.add_argument(
        "--plot-mode",
        choices=["plotly", "matplotlib"],
        default="plotly",
        help="Plotting library to use for visualizations (default: plotly)"
    )
    generate_parser.set_defaults(func=plot_model_averages)


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistics from evaluation files")
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    add_stats_subparser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
