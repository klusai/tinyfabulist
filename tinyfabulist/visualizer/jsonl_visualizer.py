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
    return pd.DataFrame(jsonl_data)

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
                style_data={'whiteSpace': 'normal'},
                style_header={
                    "fontWeight": "bold", 
                    "color": "black"
                },
                style_cell={"padding": "10px", "textAlign": "left"},
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                page_size=10,
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
    
    app = dash.Dash(__name__)
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
    df = load_data(args.input)
    app = Dash(__name__)
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
    filepath = "/home/ap/Documents/Work/Research/tiny_fabulist/tinyfabulist/data/fables/deepseek-r1-distill-llama-8b-dmb/tf_fables_deepseek-r1-distill-llama-8b-dmb_dt250305-083628.jsonl"
    main(filepath)
