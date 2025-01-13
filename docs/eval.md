https://arxiv.org/pdf/2303.16634 https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
https://arxiv.org/pdf/2303.16634

## Ensuring objectivity

# Pinning a Model Version

{
  "model": "gpt-4-2024-05-13",
  "messages": [
    {"role": "system", "content": "You are an evaluator assessing the output of another model."},
    {"role": "user", "content": "Evaluate this text: ..."}
  ]
}

