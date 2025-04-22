#!/usr/bin/env python3

import os
import json
import argparse
import glob
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def find_json_files(path):
    """Find all JSON files in a directory or return the file if it's a single file."""
    if os.path.isfile(path):
        if path.endswith('.json') or path.endswith('.jsonl'):
            return [path]
        else:
            return []
    
    json_files = []
    for extension in ['*.json', '*.jsonl']:
        json_files.extend(glob.glob(os.path.join(path, '**', extension), recursive=True))
    return json_files

def extract_model_name(file_path):
    """Extract model name from the file path."""
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    
    # Try to extract model name from filename structure
    if len(parts) >= 3:
        # This assumes a naming convention where model name is after the first underscore
        # Adjust as needed for your specific naming convention
        return parts[2]
    else:
        return file_name.split('.')[0]  # Fallback to filename without extension

def extract_metrics(file_path):
    """Extract metrics from JSON file."""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.jsonl'):
                # For JSONL, read the last line
                lines = f.readlines()
                if not lines:
                    return None
                data = json.loads(lines[-1])
            else:
                # For single JSON, load the entire file
                data = json.load(f)
                
        # Check if required metrics exist
        metrics = {}
        for metric in ['self_bleu', 'distinct_1', 'flesch_reading_ease']:
            if metric in data:
                metrics[metric] = data[metric]
            else:
                print(f"Warning: Metric '{metric}' not found in {file_path}")
                metrics[metric] = None
                
        return metrics
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def plot_metrics(metrics_by_model):
    """Create a plotly visualization for the metrics across models."""
    # Create a figure with 2 rows, 2 columns
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=("Self BLEU", "Distinct-1", "Flesch Reading Ease", "Comparison Table"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]  # "domain" type for the table
        ],
        column_widths=[0.5, 0.5],  # Equal column widths
        row_heights=[0.5, 0.5]
    )
    
    # Modern color palette - pastel tones
    colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2', '#DECF3F', '#F15854', '#4D4D4D', '#85C0F9']
    
    # Get all models
    models = list(metrics_by_model.keys())
    
    # Create a color mapping that will be consistent across charts
    model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    
    # Create table data with initial sorting by Self BLEU (first metric) for the table
    table_data = []
    model_metrics = []
    
    for model, metrics in metrics_by_model.items():
        # Format the values to 3 decimal places if they're floats
        self_bleu = metrics.get('self_bleu', 'N/A')
        distinct_1 = metrics.get('distinct_1', 'N/A')
        flesch_reading_ease = metrics.get('flesch_reading_ease', 'N/A')
        
        # Store original values for sorting
        raw_self_bleu = self_bleu if self_bleu != 'N/A' else -float('inf')
        raw_distinct_1 = distinct_1 if distinct_1 != 'N/A' else -float('inf')
        raw_flesch_reading_ease = flesch_reading_ease if flesch_reading_ease != 'N/A' else -float('inf')
        
        # Format float values to 3 decimal places for display
        if isinstance(self_bleu, float):
            self_bleu_display = f"{self_bleu:.3f}"
        else:
            self_bleu_display = self_bleu
            
        if isinstance(distinct_1, float):
            distinct_1_display = f"{distinct_1:.3f}"
        else:
            distinct_1_display = distinct_1
            
        if isinstance(flesch_reading_ease, float):
            flesch_reading_ease_display = f"{flesch_reading_ease:.3f}"
        else:
            flesch_reading_ease_display = flesch_reading_ease
        
        model_metrics.append({
            'model': model,
            'self_bleu': raw_self_bleu,
            'distinct_1': raw_distinct_1,
            'flesch_reading_ease': raw_flesch_reading_ease,
            'display': [
                model,
                self_bleu_display,
                distinct_1_display,
                flesch_reading_ease_display
            ]
        })
    
    # Sort by Self BLEU for the table (default sort)
    model_metrics.sort(key=lambda x: x['self_bleu'], reverse=True)
    
    # Generate the table data from sorted metrics
    for metric in model_metrics:
        table_data.append(metric['display'])
    
    # Add comparison table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Model', 'Self BLEU', 'Distinct-1', 'Flesch Reading Ease'],
                font=dict(size=14, color='white'),
                fill_color='#4A6B8A',  # More muted blue
                align='center',
                height=30
            ),
            cells=dict(
                values=list(zip(*table_data)),
                font=dict(size=12),
                align='center',
                height=25,
                fill_color=[['#F5F5F5' if i % 2 == 0 else 'white' for i in range(len(table_data))]]
            ),
            columnwidth=[5, 1.5, 1.5, 1.5]  # Make model column even wider
        ),
        row=2, col=2
    )
    
    # Add bar charts for each metric
    for idx, (metric_name, row, col) in enumerate([
        ('self_bleu', 1, 1),
        ('distinct_1', 1, 2),
        ('flesch_reading_ease', 2, 1)
    ]):
        # Sort models by this specific metric (descending)
        sorted_data = []
        for model, metrics in metrics_by_model.items():
            value = metrics.get(metric_name, None)
            if value is not None:
                sorted_data.append((model, value))
            else:
                # Put missing values at the end
                sorted_data.append((model, -float('inf')))
        
        # Sort by value, descending order
        sorted_data.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted models and values
        sorted_models, sorted_values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Get colors in same order as the sorted models
        sorted_colors = [model_color_map[model] if metrics_by_model[model].get(metric_name) is not None else '#CCCCCC' 
                         for model in sorted_models]
        
        # Format values for display
        text_values = [f"{v:.3f}" if isinstance(v, float) and v != -float('inf') else 'N/A' 
                       for v in sorted_values]
        
        # Replace -inf with 0 for plotting
        plot_values = [v if v != -float('inf') else 0 for v in sorted_values]
        
        # Create a single bar trace for this metric with all models
        fig.add_trace(
            go.Bar(
                x=list(sorted_models),
                y=plot_values,
                name=metric_name,
                marker_color=sorted_colors,
                text=text_values,
                textposition='auto'
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Metrics Comparison Across Models",
            'font': {'size': 24, 'color': '#333333'}
        },
        height=900,
        # width set to 100% by config parameter when saving
        showlegend=False,
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='rgba(250,250,250,0.9)',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Update axes properties
    for i in range(1, 3):
        for j in range(1, 3):
            if not (i == 2 and j == 2):  # Skip the table cell
                fig.update_xaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='#E5E5E5',
                    tickangle=45,  # Angle the model names for better readability
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='#E5E5E5',
                    row=i, col=j
                )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Generate plots of metrics from JSON files.')
    parser.add_argument('path', help='Path to a JSON file or directory containing JSON files')
    parser.add_argument('--output', help='Output file path for the plot (HTML)', default='data/metrics_plot.html')
    parser.add_argument('--csv', help='Output file path for CSV comparison', default='metrics_comparison.csv')
    args = parser.parse_args()
    
    # Find all JSON files
    json_files = find_json_files(args.path)
    
    if not json_files:
        print(f"No JSON files found in {args.path}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    metrics_by_model = defaultdict(dict)
    for file_path in json_files:
        model_name = extract_model_name(file_path)
        metrics = extract_metrics(file_path)
        
        if metrics:
            metrics_by_model[model_name] = metrics
    
    if not metrics_by_model:
        print("No valid metrics found in any files")
        return
    
    # Create and save the comparison CSV
    df = pd.DataFrame([
        {
            'model': model,
            'self_bleu': metrics.get('self_bleu', 'N/A'),
            'distinct_1': metrics.get('distinct_1', 'N/A'),
            'flesch_reading_ease': metrics.get('flesch_reading_ease', 'N/A')
        } for model, metrics in metrics_by_model.items()
    ])
    df.to_csv(args.csv, index=False)
    print(f"Comparison CSV saved to {args.csv}")
    
    # Create and save the plot
    fig = plot_metrics(metrics_by_model)
    
    # Make the figure responsive to use 80vw width
    with open(args.output, 'w') as f:
        f.write(f'''
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style>
                .plotly-graph-div {{
                    width: 100vw !important;
                    height: 100vh !important;
                    margin: 0 auto;
                }}
            </style>
        </head>
        <body>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
        </html>
        ''')
    
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()