import yaml
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
DEEPL_SCORE = 8.47
GEMMA3_12B_SCORE = 9.11

def main(file_path="tinyfabulist/estimate/config.yaml", CACHING_RATE=0.1):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        console.print(f"[bold green]Loaded configuration from[/bold green] [cyan]{file_path}[/cyan]")

    TOTAL_WORDS = config["TOTAL_WORDS"]
    TOKEN_TO_WORDS_RATIO = config["TOKEN_TO_WORDS_RATIO"]
    TOTAL_INPUT_TOKENS = TOTAL_WORDS / TOKEN_TO_WORDS_RATIO

    UNCACHED_INPUT_TOKENS = TOTAL_INPUT_TOKENS * (1 - CACHING_RATE)
    CACHED_INPUT_TOKENS = TOTAL_INPUT_TOKENS * CACHING_RATE

    ROMANIAN_TO_ENGLISH_RATIO = config["ROMANIAN_TO_ENGLISH_RATIO"]
    TOTAL_WORDS_IN_ROMANIAN = TOTAL_WORDS * ROMANIAN_TO_ENGLISH_RATIO
    TOTAL_OUTPUT_TOKENS = TOTAL_WORDS_IN_ROMANIAN / TOKEN_TO_WORDS_RATIO

    console.print(f"[bold]Total words(in english):[/bold] [yellow]{TOTAL_WORDS:,}[/yellow]")
    console.print(f"[bold]Total input tokens:[/bold] ~[yellow]{TOTAL_INPUT_TOKENS:,.2f}[/yellow]")
    console.print(f"[bold]Total uncached input tokens:[/bold] ~[yellow]{UNCACHED_INPUT_TOKENS:,.2f}[/yellow]")
    console.print(f"[bold]Total cached input tokens:[/bold] ~[yellow]{CACHED_INPUT_TOKENS:,.2f}[/yellow]")
    print()
    console.print(f"[bold]Estimated translation number of words(to romanian):[/bold] [yellow]{TOTAL_WORDS_IN_ROMANIAN:,}[/yellow]")
    console.print(f"[bold]Total output tokens:[/bold] ~[yellow]{TOTAL_OUTPUT_TOKENS:,.2f}[/yellow]")
    console.print()

    # Create a table for model comparisons
    table = Table(title="Model Cost Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Input Cost", style="green")
    table.add_column("Cached Input Cost", style="blue")
    table.add_column("Output Cost", style="magenta")
    table.add_column("Estimated Total Cost", style="red")
    table.add_column("Score", style="yellow")
    table.add_column("Cost/Score", style="bright_green")

    # load each model
    for model in config["models"]:
        name = model["name"]
        input_cost_per_million_tokens = model["input_cost_per_million_tokens"]
        cached_input_cost_per_million_tokens = model["cached_input_cost_per_million_tokens"]
        output_cost_per_million_tokens = model["output_cost_per_million_tokens"]
        score = model.get("score", "-")  # Default to "-" if no score provided

        estimated_total_cost = input_cost_per_million_tokens * UNCACHED_INPUT_TOKENS + cached_input_cost_per_million_tokens * CACHED_INPUT_TOKENS + output_cost_per_million_tokens * TOTAL_OUTPUT_TOKENS
        estimated_total_cost = estimated_total_cost / 1000000
        
        # Calculate cost/score ratio if score is available
        cost_score_ratio = "-"
        if score != "-" and estimated_total_cost > 0:
            try:
                score_value = float(score)
                cost_score_ratio = f"{estimated_total_cost/score_value:.2f}"
            except (ValueError, TypeError):
                cost_score_ratio = "-"
        
        # Add row to table
        table.add_row(
            name,
            f"${input_cost_per_million_tokens} / 1M",
            f"${cached_input_cost_per_million_tokens} / 1M",
            f"${output_cost_per_million_tokens} / 1M",
            f"~${round(estimated_total_cost)}",
            f"{score}" if score != "-" else "-",
            cost_score_ratio
        )

    ##### DEEPL #####
    deepl_cost_per_million_words = config["DEEPL_COST_PER_MILLION_WORDS"]
    total_chars = config["TOTAL_CHARS"]
    deepl_estimated_total_cost = deepl_cost_per_million_words * total_chars / 1000000

    table.add_row(
        "DeepL",
        f"${deepl_cost_per_million_words} / 1M Chars",
        f"-",
        f"-",
        f"~${round(deepl_estimated_total_cost)}",
        f"{DEEPL_SCORE}",
        f"{deepl_estimated_total_cost/DEEPL_SCORE:.2f}" if DEEPL_SCORE != "-" else "-"
    )

    # # Add Llama model
    # llama_cost = 22000
    # llama_score = LLAMA_70B_SCORE
    
    # cost_score_ratio = "-"
    # if llama_score != "-" and llama_cost > 0:
    #     try:
    #         llama_score_value = float(llama_score)
    #         cost_score_ratio = f"{llama_cost/llama_score_value:.2f}"
    #     except (ValueError, TypeError):
    #         cost_score_ratio = "-"
    
    # table.add_row(
    #     "Llama 3.3 70B",
    #     f"-",
    #     f"-",
    #     f"-",
    #     f"~${llama_cost}",
    #     f"{llama_score}" if llama_score != "-" else "-",
    #     cost_score_ratio
    # )

    # Add Gemma model
    # with L40S
    # gemma_cost = 4000
    # with L4
    #gemma_cost = 2700
    # with H200
    gemma_cost = 1300

    table.add_row(
        "Gemma 3 12B",
        f"-",
        f"-",
        f"-",
        f"~${gemma_cost}",
        f"{GEMMA3_12B_SCORE}" if GEMMA3_12B_SCORE != "-" else "-",
        f"{gemma_cost/GEMMA3_12B_SCORE:.2f}" if GEMMA3_12B_SCORE != "-" else "-"
    )

    euro_llm_cost = 675
    euro_llm_score = 8.92

    table.add_row(
        "EuroLLM 9B",
        f"-",
        f"-",
        f"-",
        f"~${euro_llm_cost}",
        f"{euro_llm_score}" if euro_llm_score != "-" else "-",
        f"{euro_llm_cost/euro_llm_score:.2f}" if euro_llm_cost != "-" else "-"
    )


    console.print(table)

if __name__ == "__main__":
    main()
