"""
Cost estimator for different OpenAI models (GPT-4o and GPT-o3-mini)
"""

# Model pricing in USD per 1M tokens
models = {
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
        "name": "GPT-4o"
    },
    "gpt-o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
        "name": "GPT-o3-mini"
    }
}

one_mil = 1_000_000

# Assumed tokens per request/line
input_tokens = {
    "median": 350, # Median estimate
    "max": 450     # Q3 estimate
}

output_tokens = {
    "median": 420,
    "max": 540 #20% increase in romanian
}

# Number of requests/lines (2.5 million)
num_requests = 2.5 * one_mil

# Cache rate (percentage of inputs that are cached)
cache_rate = 0.25  # 25% of inputs are cached

def calculate_costs(model_id, num_requests, cache_rate=0.25):
    """
    Calculate costs for a specific model and print detailed breakdown.

    Returns:
        A dictionary of cost estimates for each scenario.
    """
    model = models[model_id]
    results = {}
    
    print(f"\n== Cost Estimation for {model['name']} ==")
    print(f"Number of requests: {num_requests:,.0f}")
    print(f"Cache rate: {cache_rate:.1%}\n")
    
    # Calculate for different token scenarios: median and max only
    for input_scenario in ["median", "max"]:
        for output_scenario in ["median", "max"]:
            # Calculate tokens used for inputs and outputs
            regular_input_tokens = (1 - cache_rate) * num_requests * input_tokens[input_scenario]
            cached_input_tokens = cache_rate * num_requests * input_tokens[input_scenario]
            total_output_tokens = num_requests * output_tokens[output_scenario]
            
            # Calculate costs (price per million tokens)
            regular_input_cost = (regular_input_tokens / one_mil) * model["input"]
            cached_input_cost = (cached_input_tokens / one_mil) * model["cached_input"]
            output_cost = (total_output_tokens / one_mil) * model["output"]
            
            # Total cost for the scenario
            total_cost = regular_input_cost + cached_input_cost + output_cost
            
            scenario_name = (f"{input_scenario.capitalize()} input ({input_tokens[input_scenario]} tokens) + "
                             f"{output_scenario.capitalize()} output ({output_tokens[output_scenario]} tokens)")
            results[scenario_name] = {
                "input_tokens": regular_input_tokens + cached_input_tokens,
                "output_tokens": total_output_tokens,
                "input_cost": regular_input_cost + cached_input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
    
    # Print detailed results for each scenario
    for scenario, data in results.items():
        print(f"Scenario: {scenario}")
        print(f"  Input Cost:  ${data['input_cost']:,.2f} ({data['input_tokens']:,.0f} tokens)")
        print(f"  Output Cost: ${data['output_cost']:,.2f} ({data['output_tokens']:,.0f} tokens)")
        print(f"  Total Cost:  ${data['total_cost']:,.2f}\n")
    
    # Print summary for the median scenario
    median_scenario = "Median input (350 tokens) + Median output (420 tokens)"
    if median_scenario in results:
        summary = results[median_scenario]
        print("== SUMMARY (Median Scenario) ==")
        print(f"Total estimated cost for {model['name']}: ${summary['total_cost']:,.2f}")
        print(f"Cost per request: ${summary['total_cost'] / num_requests:.5f}\n")
    else:
        print("Median scenario data not found.\n")
    
    return results

def main():
    """Main function that runs cost calculations for all models and prints a comparison."""
    median_summary = {}
    max_summary = {}
    
    # Calculate and collect results for each model
    for model_id in models:
        results = calculate_costs(model_id, num_requests, cache_rate)
        median_key = "Median input (350 tokens) + Median output (420 tokens)"
        max_key = "Max input (450 tokens) + Max output (540 tokens)"
        if median_key in results:
            median_summary[model_id] = results[median_key]
        if max_key in results:
            max_summary[model_id] = results[max_key]
    
    # Print side-by-side model comparison for the median scenario
    print("== MODEL COMPARISON (Median Scenario) ==")
    for model_id, data in median_summary.items():
        model_name = models[model_id]["name"]
        total_cost = data["total_cost"]
        cost_per_req = total_cost / num_requests
        print(f"{model_name} (Median):")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Cost per request: ${cost_per_req:.5f}\n")
    
    # Print side-by-side model comparison for the max scenario
    print("== MODEL COMPARISON (Max Scenario) ==")
    for model_id, data in max_summary.items():
        model_name = models[model_id]["name"]
        total_cost = data["total_cost"]
        cost_per_req = total_cost / num_requests
        print(f"{model_name} (Max):")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Cost per request: ${cost_per_req:.5f}\n")

if __name__ == "__main__":
    main()
