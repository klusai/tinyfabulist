import os

# List of prompts and model responses
data = [
    {
        "prompt": "Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room. As Lily was decorating her room, the sky outside became dark. There was a loud ***",
        "completion": "thunderclap and a bright flash of lightning. Lily was a little scared, but she knew she had to be brave. She told her mom and dad, and they all went outside to see what was going on."
    },
    {
        "prompt": "In a small village, there was a young boy who dreamed of flying. One day, he discovered an old set of wings in his attic. He ***",
        "completion": "decided to repair them. After weeks of hard work, he managed to build a working pair of wings. With a leap of faith, he soared into the sky."
    },
]

# Loop through each pair of prompt and response
for i, pair in enumerate(data):
    prompt = pair["prompt"]
    completion = pair["completion"]
    output_file = f"evaluation_results_{i+1}.json"
    
    # Run evals_cli.py for each pair
    os.system(f'python evals_cli.py --prompt "{prompt}" --completion "{completion}" --output {output_file}')
