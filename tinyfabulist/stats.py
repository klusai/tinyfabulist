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
            if "Evaluation" in base_name:
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
                moral = scores["moral_clarity"] / count
                translation_averages[model] = {
                    "translation_accuracy": ta,
                    "fluency": fluency,
                    "moral_clarity": moral,
                    "average_score_(mean)": (ta + fluency + moral) / 3,
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
            "| Model | Translation Accuracy | Fluency | Moral Clarity | Average Score (Mean) | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) |",
            "|-------|----------------------|---------|---------------|----------------------|-------|------------------|-------------------|------------------------|",
        ]
        rows = []
        for model, metrics in translation_averages.items():
            row = (f"| {model} | {metrics['translation_accuracy']:.2f} | {metrics['fluency']:.2f} | "
                   f"{metrics['moral_clarity']:.2f} | "
                   f"{metrics['average_score_(mean)']:.2f} | {metrics['count']} | "
                   f"{metrics['input_tokens']:.1f} | {metrics['output_tokens']:.1f} | "
                   f"{metrics['inference_time']:.2f} |")
            rows.append(row)
        return "\n".join(header + [""] + rows)

    @staticmethod
    def count_age_groups(evaluation_files):
        """Count the frequency of each age group from evaluation files."""
        age_group_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        total_evals = 0
        
        for eval_file in evaluation_files:
            try:
                with open(eval_file, "r") as f:
                    for line in f:
                        try:
                            eval_data = json.loads(line.strip())
                            if "best_age_group" in eval_data["evaluation"]:
                                age_group = eval_data["evaluation"]["best_age_group"]
                                if age_group in age_group_counts:
                                    age_group_counts[age_group] += 1
                                total_evals += 1
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line in {eval_file}")
            except Exception as e:
                logger.error(f"Error processing {eval_file}: {e}")
        
        # Convert to percentages
        age_group_percentages = {
            group: (count / total_evals * 100 if total_evals > 0 else 0) 
            for group, count in age_group_counts.items()
        }
        
        return age_group_counts, age_group_percentages, total_evals

    @staticmethod
    def count_age_groups_by_file(evaluation_files):
        """Count the frequency of each age group for each evaluation file."""
        results = {}
        model_name = ""
        
        for eval_file in evaluation_files:
            file_name = os.path.basename(eval_file)
            age_group_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
            total_evals = 0
            
            try:
                with open(eval_file, "r") as f:
                    for line in f:
                        try:
                            eval_data = json.loads(line.strip())
                            model_name = eval_data["llm_name"]
                            if "evaluation" in eval_data and "best_age_group" in eval_data["evaluation"]:
                                age_group = eval_data["evaluation"]["best_age_group"]
                                if age_group in age_group_counts:
                                    age_group_counts[age_group] += 1
                                    total_evals += 1
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line in {file_name}")
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
            
            if total_evals > 0:
                # Store results for this file
                results[model_name] = {
                    "counts": age_group_counts,
                    "total": total_evals,
                    "file": file_name
                }
        
        return results


# --- Visualization Module ---
class Visualizer:
    # ----- Plotly Visualizations -----
    @staticmethod
    def generate_plotly_visualizations(averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str, orientation: str = "vertical"):
        models_list = list(averages.keys())
        metrics_list = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt", "average_score_(mean)"]
        fig = Visualizer.create_plotly_figure(averages, tf_model_stats, metrics_list, models_list, orientation)

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
    def create_plotly_figure(averages: dict, tf_model_stats: dict, metrics_list: list, models_list: list, orientation: str = "vertical") -> go.Figure:
        # Sort models for better visualization
        models_list = sorted(models_list, key=lambda x: -averages[x]['average_score_(mean)'])
        
        if orientation == "horizontal":
            # Use existing horizontal layout with custom row heights
            fig = make_subplots(
                rows=2, cols=1,
                vertical_spacing=0.1,
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],  # Top row: evaluation metrics split in two
                    [{"type": "bar", "colspan": 2}, None],  # Bottom row: performance metrics spanning both columns
                ],
                row_heights=[0.75, 0.25]  # First plot takes 75% of height
            )
        else:
            # For vertical orientation, split models into two columns for better readability
            fig = make_subplots(
                rows=3, cols=1,
                vertical_spacing=0.2,
                horizontal_spacing=0.1,
                specs=[
                    [{"type": "bar"}], 
                    [{"type": "bar"}],
                    [{"type": "bar"}],  # Bottom row: performance metrics spanning both columns
                ],
            )
        
        # Define colors for evaluation metrics
        eval_colors = {
            "grammar": "#1DB954",
            "creativity": "#191414", 
            "moral_clarity": "#535353",
            "adherence_to_prompt": "#B3B3B3",
            "average_score_(mean)": "#E60012",
        }
        
        # Split models into two groups if in vertical orientation
        if orientation == "vertical" and len(models_list) > 5:
            mid_point = len(models_list) // 2
            models_left = models_list[:mid_point]
            models_right = models_list[mid_point:]
        else:
            models_left = models_list
            models_right = []
        
        # Add evaluation metric traces
        for i, metric in enumerate(metrics_list):
            if orientation == "horizontal":
                # Horizontal layout (single column)
                values = [averages[model][metric] for model in models_list]
                fig.add_trace(
                    go.Bar(
                        y=models_list,
                        x=values,
                        name=metric.replace("_", " ").title(),
                        text=[f"{val:.2f}" for val in values],
                        textposition="auto",
                        marker_color=eval_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{y}}<br>{metric.replace("_", " ").title()}: %{{x:.2f}}<extra></extra>',
                        orientation='h',
                    ),
                    row=1, col=1
                )
            else:
                # Vertical layout (split into two columns)
                # Left column
                values_left = [averages[model][metric] for model in models_left]
                fig.add_trace(
                    go.Bar(
                        x=models_left,
                        y=values_left,
                        name=metric.replace("_", " ").title(),
                        text=[f"{val:.2f}" for val in values_left],
                        textposition="auto",
                        marker_color=eval_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
                        showlegend=True,
                    ),
                    row=1, col=1
                )
                
                # Right column (if models are split)
                if models_right:
                    values_right = [averages[model][metric] for model in models_right]
                    fig.add_trace(
                        go.Bar(
                            x=models_right,
                            y=values_right,
                            name=metric.replace("_", " ").title(),
                            text=[f"{val:.2f}" for val in values_right],
                            textposition="auto",
                            marker_color=eval_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                            hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
                            showlegend=False  # Don't duplicate legends
                        ),
                        row=2, col=1
                    )
        
        # Performance metrics (bottom row)
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
            if orientation == "horizontal":
                values = [source.get(model, {}).get(metric, 0) for model in models_list]
                fig.add_trace(
                    go.Bar(
                        y=models_list,
                        x=values,
                        name=metric.replace("avg_", "").replace("_", " ").title(),
                        text=[f"{val:.1f}" for val in values],
                        textposition="auto",
                        marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{y}}<br>{metric.replace("avg_", "").replace("_", " ").title()}: %{{x:.1f}}<extra></extra>',
                        orientation='h',
                    ),
                    row=2, col=1
                )
            else:
                values = [source.get(model, {}).get(metric, 0) for model in models_list]
                fig.add_trace(
                    go.Bar(
                        x=models_list,
                        y=values,
                        name=metric.replace("avg_", "").replace("_", " ").title(),
                        text=[f"{val:.1f}" for val in values],
                        textposition="auto",
                        marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f"%{{x}}<br>{metric.replace('avg_', '').replace('_', ' ').title()}: %{{y:.1f}}<extra></extra>",
                    ),
                    row=3, col=1
                )
        
        # Configure the layout based on orientation
        if orientation == "horizontal":
            fig.update_layout(height=3000, margin=dict(l=250, r=80, t=120, b=80))
        else:
            # For vertical layout, adjust height based on number of models
            height = 1200 if len(models_list) <= 10 else 1800
            fig.update_layout(height=height)
            
            # Set consistent y-axis range for evaluation metrics
            fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
            if models_right:
                fig.update_yaxes(title_text="", range=[0, 10], row=1, col=2)
 
        Visualizer.configure_plotly_layout(fig, orientation)
        return fig

    @staticmethod
    def configure_plotly_layout(fig: go.Figure, orientation: str = "vertical"):
        # Determine if we have a split layout by checking column count in row 1
        split_layout = len(fig._grid_ref) > 2 and orientation == "vertical"
        
        fig.update_layout(
            title_text="Model Evaluation and Performance Analytics",
            title_x=0.5,
            title_font=dict(size=24),
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.00,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
                traceorder="normal"
            ),
            template="plotly_white",
            width=1920,
            margin=dict(
                l=80,
                r=80,
                t=140 if split_layout else 120,
                b=100 if split_layout else 80
            ),
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
        )
        
        if orientation == "horizontal":
            # Horizontal layout axis configuration
            fig.update_xaxes(row=1, col=1, title_text="Score", range=[0, 10], 
                        title_font=dict(size=16), tickfont=dict(size=14), automargin=True)
            fig.update_xaxes(row=2, col=1, title_text="Value", 
                        title_font=dict(size=16), tickfont=dict(size=14), automargin=True)
            # Ensure y-axis labels (model names) have enough space
            fig.update_yaxes(row=1, col=1, tickfont=dict(size=14), automargin=True)
            fig.update_yaxes(row=2, col=1, tickfont=dict(size=14), automargin=True)
        else:
            # Vertical layout - already configured in main function
            # Just ensure consistent tick formatting and rotation
            if split_layout:
                # Ensure x-axis labels are rotated for better readability in split layout
                fig.update_xaxes(tickangle=45, row=1, col=1, automargin=True)
                fig.update_xaxes(tickangle=45, row=1, col=2, automargin=True)
                fig.update_xaxes(tickangle=45, row=2, col=1, automargin=True)
            else:
                fig.update_xaxes(tickangle=45, automargin=True)

    @staticmethod
    def generate_translation_plotly_visualizations(translation_averages: dict, tf_model_stats: dict, stats_folder: str, timestamp: str, output_mode: str, orientation: str = "vertical"):
        models_list = list(translation_averages.keys())
        metrics_list = ["translation_accuracy", "fluency", "moral_clarity", "average_score_(mean)"]

        # Sort models for better visualization
        models_list = sorted(models_list, key=lambda x: -translation_averages[x]['average_score_(mean)'])
        
        if orientation == "horizontal":
            # Use horizontal layout with custom row heights
            fig = make_subplots(
                rows=2, cols=1,
                vertical_spacing=0.1,
                specs=[[{"type": "bar"}], [{"type": "bar"}]],
                row_heights=[0.75, 0.25]  # First plot takes 75% of height
            )
        else:
            # For vertical orientation, split models into TWO ROWS (one under the other)
            # for better readability instead of side-by-side columns
            if len(models_list) > 5:
                # Create a 3-row layout: first half of models, second half of models, performance metrics
                fig = make_subplots(
                    rows=3, cols=1,
                    vertical_spacing=0.12,
                    specs=[
                        [{"type": "bar"}],  # First row: first half of models
                        [{"type": "bar"}],  # Second row: second half of models
                        [{"type": "bar"}],  # Third row: performance metrics
                    ],
                    row_heights=[0.4, 0.4, 0.2]  # Split evaluation metrics equally
                )
            else:
                # If few models, just use two rows
                fig = make_subplots(
                    rows=2, cols=1,
                    vertical_spacing=0.15,
                    specs=[
                        [{"type": "bar"}],  # First row: evaluation metrics
                        [{"type": "bar"}],  # Second row: performance metrics
                    ],
                    row_heights=[0.7, 0.3]
                )
        
        trans_colors = {
            "translation_accuracy": "#1DB954",
            "fluency": "#191414",
            "moral_clarity": "#B3B3B3",
            "average_score_(mean)": "#E60012",
        }
        
        # Split models into two groups if in vertical orientation with many models
        if orientation == "vertical" and len(models_list) > 5:
            mid_point = len(models_list) // 2
            models_top = models_list[:mid_point]
            models_bottom = models_list[mid_point:]
            models_top_names = [model.split("_")[-1] for model in models_top]
            models_bottom_names = [model.split("_")[-1] for model in models_bottom]
        else:
            models_top = models_list
            models_top_names = [model.split("_")[-1] for model in models_top]
            models_bottom = []
            models_bottom_names = []
        
        # Add evaluation metric traces
        for i, metric in enumerate(metrics_list):
            if orientation == "horizontal":
                values = [translation_averages[model][metric] for model in models_list]
                fig.add_trace(
                    go.Bar(
                        y=models_list,
                        x=values,
                        name=metric.replace("_", " ").title(),
                        text=[f"{val:.2f}" for val in values],
                        textposition="auto",
                        marker_color=trans_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{y}}<br>{metric.replace("_", " ").title()}: %{{x:.2f}}<extra></extra>',
                        orientation='h',
                    ),
                    row=1, col=1
                )
            else:
                # Top group of models (row 1)
                values_top = [translation_averages[model][metric] for model in models_top]
                fig.add_trace(
                    go.Bar(
                        x=models_top_names,
                        y=values_top,
                        name=metric.replace("_", " ").title(),
                        text=[f"{val:.2f}" for val in values_top],
                        textposition="auto",
                        marker_color=trans_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
                        showlegend=True  # Show legend for the first group
                    ),
                    row=1, col=1
                )
                
                # Bottom group of models (row 2 if split)
                if models_bottom:
                    values_bottom = [translation_averages[model][metric] for model in models_bottom]
                    fig.add_trace(
                        go.Bar(
                            x=models_bottom_names,
                            y=values_bottom,
                            name=metric.replace("_", " ").title(),
                            text=[f"{val:.2f}" for val in values_bottom],
                            textposition="auto",
                            marker_color=trans_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                            hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
                            showlegend=False  # Don't duplicate legends for second group
                        ),
                        row=2, col=1
                    )
        
        # Performance metrics for bottom row
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
    
        performance_row = 2 if len(models_bottom) == 0 else 3
        
        for i, metric in enumerate(performance_metrics):
            if orientation == "horizontal":
                values = [source.get(model, {}).get(metric, 0) for model in models_list]
                fig.add_trace(
                    go.Bar(
                        y=models_list,
                        x=values,
                        name=metric.replace("avg_", "").replace("_", " ").title(),
                        text=[f"{val:.1f}" for val in values],
                        textposition="auto",
                        marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f'%{{y}}<br>{metric.replace("avg_", "").replace("_", " ").title()}: %{{x:.1f}}<extra></extra>',
                        orientation='h',
                    ),
                    row=2, col=1
                )
            else:
                values = [source.get(model, {}).get(metric, 0) for model in models_list]
                model_names = [model.split("_")[-1] for model in models_list]
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=values,
                        name=metric.replace("avg_", "").replace("_", " ").title(),
                        text=[f"{val:.1f}" for val in values],
                        textposition="auto",
                        marker_color=perf_colors.get(metric.replace("avg_", ""), f"hsl({50 + i * 70}, 70%, 50%)"),
                        hovertemplate=f"%{{x}}<br>{metric.replace('avg_', '').replace('_', ' ').title()}: %{{y:.1f}}<extra></extra>",
                    ),
                    row=performance_row, col=1
                )
        
        # Configure the layout based on orientation
        if orientation == "horizontal":
            height = 3000
            left_margin = 250
        else:
            # For vertical layout with models in two rows
            height = 2000 if len(models_list) > 10 else 1600
            left_margin = 80
            
            # Set consistent y-axis range for translation evaluation metrics
            fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
            if models_bottom:
                fig.update_yaxes(title_text="Score", range=[0, 10], row=2, col=1)
                
            # Add annotations to clarify which models are in each row
            if models_bottom:
                fig.add_annotation(
                    x=0.5, y=0.99,
                    xref="paper", yref="paper",
                    text=f"First Group (Models 1-{len(models_top)})",
                    showarrow=False,
                    font=dict(size=14)
                )
                fig.add_annotation(
                    x=0.5, y=0.59,
                    xref="paper", yref="paper",
                    text=f"Second Group (Models {len(models_top)+1}-{len(models_list)})",
                    showarrow=False,
                    font=dict(size=14)
                )

        # Apply common layout settings
        fig.update_layout(
            title_text="Translation Evaluation and Performance Analytics",
            title_x=0.5,
            title_font=dict(size=24),
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
                traceorder="normal"
            ),
            template="plotly_white",
            height=height,
            width=1920,
            margin=dict(l=left_margin, r=80, t=120, b=80),
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
        )
        
        # Configure axes
        if orientation == "horizontal":
            # For horizontal bars
            fig.update_xaxes(row=1, col=1, title_text="Score", range=[0, 10], 
                        title_font=dict(size=16), tickfont=dict(size=14), automargin=True)
            fig.update_xaxes(row=2, col=1, title_text="Value", 
                        title_font=dict(size=16), tickfont=dict(size=14), automargin=True)
            fig.update_yaxes(row=1, col=1, tickfont=dict(size=14), automargin=True)
            fig.update_yaxes(row=2, col=1, tickfont=dict(size=14), automargin=True)
        else:
            # For vertical layout with rows instead of columns
            fig.update_xaxes(tickangle=45, row=1, col=1, automargin=True)
            if models_bottom:
                fig.update_xaxes(tickangle=45, row=2, col=1, automargin=True)
                fig.update_xaxes(tickangle=45, row=3, col=1, automargin=True)
                fig.update_yaxes(title_text="Performance", row=3, col=1)
            else:
                fig.update_xaxes(tickangle=45, row=2, col=1, automargin=True)
                fig.update_yaxes(title_text="Performance", row=2, col=1)

        # Save outputs
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
        metrics_list = ["translation_accuracy", "fluency", "moral_clarity", "average_score_(mean)"]
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

    @staticmethod
    def generate_age_group_plot(age_group_data, stats_folder, timestamp, output_mode):
        """Generate a plot showing age group distribution."""
        counts, percentages, total = age_group_data
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Age group descriptions
        age_descriptions = {
            "A": "3 years or under",
            "B": "4-7 years",
            "C": "8-11 years",
            "D": "12-15 years",
            "E": "16 years or above"
        }
        
        # Create x-axis labels with descriptions
        x_labels = [f"{group} ({age_descriptions[group]})" for group in counts.keys()]
        
        # Add count bars
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=list(counts.values()),
                name="Count",
                text=[f"{count}" for count in counts.values()],
                textposition="auto",
                marker_color="#1DB954",
                hovertemplate="Age Group: %{x}<br>Count: %{y}<extra></extra>",
            ),
            secondary_y=False,
        )
        
        # Add percentage line
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=list(percentages.values()),
                name="Percentage",
                mode="lines+markers+text",
                text=[f"{p:.1f}%" for p in percentages.values()],
                textposition="top center",
                marker=dict(size=10, color="#E60012"),
                line=dict(width=3, color="#E60012"),
                hovertemplate="Age Group: %{x}<br>Percentage: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Age Group Distribution (Total: {total} Evaluations)",
            title_x=0.5,
            title_font=dict(size=24),
            title_font_color="#333333",
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
            ),
            template="plotly_white",
            height=800,
            width=1200,
            margin=dict(l=80, r=80, t=120, b=80),
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
        )
        
        # Update yaxes
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Percentage (%)", secondary_y=True)
        
        # Save outputs if required
        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_age_group_plot_{timestamp}.png")
            html_filename = os.path.join(stats_folder, f"tf_stats_age_group_plot_{timestamp}.html")
            fig.write_image(plot_filename)
            fig.write_html(html_filename)
            logger.info(f"Age group distribution plot saved to {plot_filename} and {html_filename}")
        if output_mode in ["terminal", "both"]:
            fig.show()

        return fig

    @staticmethod
    def generate_age_groups_by_file_plot(age_groups_by_file, stats_folder, timestamp, output_mode):
        """Generate a stacked bar chart showing age group distribution by file/model."""
        if not age_groups_by_file:
            logger.info("No age group data by file available")
            return
        
        # Age group descriptions for hover info
        age_descriptions = {
            "A": "3 years or under",
            "B": "4-7 years",
            "C": "8-11 years",
            "D": "12-15 years",
            "E": "16 years or above"
        }
        
        # Create figure
        fig = go.Figure()
        
        # Sort files by name for consistent display
        models = sorted(age_groups_by_file.keys())
        
        # Define colors for age groups
        colors = {
            "A": "#FF9999",  # Light red
            "B": "#66B2FF",  # Light blue
            "C": "#99FF99",  # Light green
            "D": "#FFCC99",  # Light orange
            "E": "#CC99FF"   # Light purple
        }
        
        # Add traces for each age group
        for age_group in ["A", "B", "C", "D", "E"]:
            percentages = []
            hover_texts = []
            
            for model in models:
                data = age_groups_by_file[model]
                count = data["counts"][age_group]
                total = data["total"]
                percentage = (count / total * 100) if total > 0 else 0
                
                percentages.append(percentage)
                hover_texts.append(
                    f"Model: {model}<br>"
                    f"Age Group: {age_group} ({age_descriptions[age_group]})<br>"
                    f"Count: {count}<br>"
                    f"Percentage: {percentage:.1f}%<br>"
                    f"Total evals: {total}"
                )
            
            fig.add_trace(go.Bar(
                x=models,
                y=percentages,
                name=f"Age {age_group} - {age_descriptions[age_group]}",
                text=[f"{p:.1f}%" for p in percentages],
                textposition="inside",
                marker_color=colors[age_group],
                hoverinfo="text",
                hovertext=hover_texts
            ))
        
        # Update layout to stacked bars
        fig.update_layout(
            title="Age Group Distribution by Model",
            title_x=0.5,
            title_font=dict(size=24),
            barmode='stack',
            xaxis=dict(title="Model", tickangle=45),
            yaxis=dict(title="Percentage (%)", range=[0, 100]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=800,
            width=1200
        )
        
        # Save outputs if required
        if output_mode in ["files", "both"]:
            plot_filename = os.path.join(stats_folder, f"tf_stats_age_groups_by_model_{timestamp}.png")
            html_filename = os.path.join(stats_folder, f"tf_stats_age_groups_by_model_{timestamp}.html")
            fig.write_image(plot_filename)
            fig.write_html(html_filename)
            logger.info(f"Age group distribution by model saved to {plot_filename} and {html_filename}")
        if output_mode in ["terminal", "both"]:
            fig.show()
            
        return fig


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


def process_eval_files(eval_files, tf_model_stats, stats_folder, timestamp, output_mode, plot_mode="plotly", orientation="vertical"):
    score_totals = Evaluator.parse_evaluation_data(eval_files)
    averages = Evaluator.compute_averages(score_totals)
    
    # Generate standard model comparison visualizations
    md_table = Evaluator.create_evaluation_markdown_table(averages)
    if output_mode in ["files", "both"]:
        md_filename = os.path.join(stats_folder, f"tf_stats_eval_table_{timestamp}.md")
        with open(md_filename, "w") as f:
            f.write(md_table)
        logger.info(f"Evaluation table saved to {md_filename}")
    if output_mode in ["terminal", "both"]:
        print("\n" + md_table + "\n")
        
    if plot_mode.lower() == "plotly":
        Visualizer.generate_plotly_visualizations(averages, tf_model_stats, stats_folder, timestamp, output_mode, orientation)
    else:
        Visualizer.generate_matplotlib_visualizations(averages, tf_model_stats, stats_folder, timestamp, output_mode)
        
    # # Generate age group distribution plot for all files combined
    # age_group_data = Evaluator.count_age_groups(eval_files)
    # if any(age_group_data[0].values()):  # Check if we found any age group data
    #     Visualizer.generate_age_group_plot(age_group_data, stats_folder, timestamp, output_mode)
        
    #     # Generate age group distribution plot by file/model
    #     age_groups_by_file = Evaluator.count_age_groups_by_file(eval_files)
    #     Visualizer.generate_age_groups_by_file_plot(age_groups_by_file, stats_folder, timestamp, output_mode)
    # else:
    #     logger.info("No age group data found in the evaluation files")
    
    # Generate age group distribution by file/model plot
    age_groups_by_file = Evaluator.count_age_groups_by_file(eval_files)
    if age_groups_by_file:
        Visualizer.generate_age_groups_by_file_plot(age_groups_by_file, stats_folder, timestamp, output_mode)
    else:
        logger.info("No age group data by file found in the evaluation files")


def process_translation_eval_files(translation_eval_files, tf_model_stats, stats_folder, timestamp, output_mode, plot_mode="plotly", orientation="vertical"):
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
        Visualizer.generate_translation_plotly_visualizations(translation_averages, tf_model_stats, stats_folder, timestamp, output_mode, orientation)
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
        process_eval_files(standard_eval_files, tf_model_stats, stats_folder, timestamp, args.output_mode, args.plot_mode, args.orientation)
    else:
        logger.info("No standard evaluation files found.")
    if translation_eval_files:
        process_translation_eval_files(translation_eval_files, tf_model_stats, stats_folder, timestamp, args.output_mode, args.plot_mode, args.orientation)
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
    generate_parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientation of bar charts: vertical (default) or horizontal"
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