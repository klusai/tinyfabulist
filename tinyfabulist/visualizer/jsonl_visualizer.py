import os
import dash
import pandas as pd
import json
from dash import html, dcc, Input, Output, no_update, dash_table

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def read_jsonl_file(filepath):
    """Reads a JSONL file and returns a list of JSON objects."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def load_data(filepath):
    """Loads JSONL data from the file and returns a Pandas DataFrame."""
    jsonl_data = read_jsonl_file(filepath)
    df = pd.DataFrame(jsonl_data)
    
    # Preprocess the dataframe to convert complex objects to strings
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x, indent=2) if isinstance(x, (dict, list)) else x)
    
    return df

# -----------------------------------------------------------------------------
# Layout Creation
# -----------------------------------------------------------------------------
def create_layout(df, filename):
    """Builds and returns the Dash app layout. CSS styling is defined externally."""
    layout = html.Div([
        html.H1("TinyFabulist JSONL Visualizer 📊", className="header-title"),
        html.P(f"Filename: {filename}", className="filename"),
        
        # Search bar to filter table rows
        dcc.Input(
            id="search-input", 
            type="text", 
            placeholder="Search...",
            className="search-input"
        ),
        
        # DataTable for displaying the data.
        html.Div(
            dash_table.DataTable(
                id='table',
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'lineHeight': '1.2',
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                },
                style_header={
                    "fontWeight": "bold", 
                    "color": "black",
                    "backgroundColor": "#f1f1f1",
                    "height": "auto",
                },
                style_cell={
                    "padding": "10px", 
                    "textAlign": "left",
                    "minWidth": "200px",
                    "width": "250px",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "normal",
                    "wordBreak": "break-word",
                },
                style_table={
                    'overflowX': 'scroll',
                    'width': '100%',
                    'minHeight': '300px',
                    'height': 'auto',
                    'maxHeight': '600px',
                },
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                page_size=10,
                css=[{"selector": "table", "rule": "table-layout: fixed;"}],
                tooltip_data=[
                    {
                        column: {'value': str(value), 'type': 'markdown'}
                        for column, value in row.items()
                    } for row in df.to_dict('records')
                ],
                tooltip_duration=None,
            ),
            className="dash-table-container"
        ),
        
        # Modal overlay for the popup.
        html.Div(
            id="modal-overlay",
            n_clicks=0,
            children=[
                html.Div(
                    id="modal-div",
                    children=[
                        # Close button using a Unicode icon (✖)
                        html.Button("✖", id="close-btn"),
                        # Copy button using the Clipboard component with a clipboard icon (📋)
                        dcc.Clipboard(
                            id="copy-to-clipboard-btn",
                            target_id="modal-content",
                            title="Copy to clipboard",
                            content="📋"
                        ),
                        # Modal content
                        html.Div(
                            id="modal-content"
                        )
                    ]
                )
            ]
        )
    ])
    return layout

# -----------------------------------------------------------------------------
# Callback Registration
# -----------------------------------------------------------------------------
def register_callbacks(app, df):
    """Registers all Dash callbacks on the provided app."""
    
    @app.callback(
        Output('table', 'data'),
        Input('search-input', 'value')
    )
    def update_table(search_value):
        if not search_value:
            return df.to_dict('records')
        
        normalized_search = "".join(search_value.split()).lower()
        filtered_df = df[df.apply(
            lambda row: row.astype(str)
            .apply(lambda x: "".join(x.split()).lower())
            .apply(lambda x: normalized_search in x)
            .any(), axis=1)]
        return filtered_df.to_dict('records')
    
    @app.callback(
        [Output("modal-content", "children"),
         Output("modal-overlay", "style")],
        [Input("table", "active_cell"),
         Input("close-btn", "n_clicks"),
         Input("modal-overlay", "n_clicks"),
         Input("modal-div", "n_clicks")],
        [dash.dependencies.State("table", "data")]
    )
    def update_modal(active_cell, close_btn_n_clicks, overlay_n_clicks, modal_div_n_clicks, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "", {"display": "none"}
        
        triggered_id = ctx.triggered[0]["prop_id"]
        
        # If the click is on modal-div itself, do nothing.
        if triggered_id.startswith("modal-div"):
            return no_update, no_update
        
        # If the close button or overlay was clicked, close the modal.
        if "close-btn" in triggered_id or "modal-overlay" in triggered_id:
            return "", {"display": "none"}
        
        # If a cell is clicked, show its content.
        if active_cell is None:
            return "", {"display": "none"}
        
        row = active_cell["row"]
        col = active_cell["column_id"]
        cell_value = data[row][col]
        
        # Format cell value if it's a JSON string
        if isinstance(cell_value, str) and (cell_value.startswith('{') or cell_value.startswith('[')):
            try:
                # Try to parse and pretty-print JSON
                parsed_json = json.loads(cell_value)
                cell_value = json.dumps(parsed_json, indent=2)
                # Wrap in pre tag for proper formatting
                cell_value = html.Pre(cell_value, style={'white-space': 'pre-wrap'})
            except:
                # If not valid JSON, just show as is
                pass
        
        # Return a modal style that centers the content.
        modal_style = {
            "display": "flex",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0, 0, 0, 0.6)",
            "zIndex": 1000,
            "justifyContent": "center",
            "alignItems": "center",
            "padding": "20px"
        }
        return cell_value, modal_style
        
# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main(filepath):
    df = load_data(filepath)
    
    app = dash.Dash(__name__, 
                   suppress_callback_exceptions=True,
                   external_stylesheets=[])
    app.layout = create_layout(df, filename=filepath)
    register_callbacks(app, df)
    app.run(debug=False)

def add_visualize_subparser(subparsers) -> None:
    """
    Add a subparser for the JSONL visualization dashboard.
    
    Args:
        subparsers: The subparsers object from argparse to add this parser to.
    """
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Launch interactive visualization dashboard for JSONL files"
    )
    
    visualize_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file to visualize"
    )
    
    visualize_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the visualization server on (default: 8050)"
    )
    
    visualize_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the visualization server in debug mode"
    )
    
    visualize_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the visualization server on (default: 127.0.0.1)"
    )
    
    visualize_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser"
    )
    
    # Set the function that will handle the visualize command
    visualize_parser.set_defaults(func=launch_visualization)


def launch_visualization(args):
    """
    Launch the JSONL visualization dashboard.
    
    Args:
        args: The parsed arguments from the command line.
    """
    from dash import Dash
    import webbrowser
    import threading
    import time
    
    # Verify the file exists
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return
    
    # Create and configure the Dash app
    df = load_data(filepath=args.input)
    
    # Handle empty dataframes
    if df.empty:
        print("Warning: The JSONL file is empty or could not be parsed properly.")
        df = pd.DataFrame({'message': ['No data found in the file.']})
    
    app = Dash(__name__, 
              suppress_callback_exceptions=True,
              external_stylesheets=[])
    app.layout = create_layout(df, filename=args.input)
    register_callbacks(app, df)
    
    # Open browser automatically unless --no-browser is specified
    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    # Run the server
    print(f"Starting visualization server for {args.input}")
    print(f"Dashboard available at http://{args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)

# -----------------------------------------------------------------------------
# Run the Application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = "/home/andrei/Documents/Work/tinyfabulist/data/evaluations/Evaluation_250526-125637.jsonl"
    main(filepath)
