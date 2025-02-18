# Fable Evaluation and Diversity Assessment

## **1. Introduction**
This project automates the evaluation of AI-generated fables using **GPT-4-Turbo (2024-01-25)**. The system assesses grammar, creativity, and consistency while estimating the author's age group. Additionally, it evaluates dataset diversity based on vocabulary, syntax, themes, and uniqueness.

## **2. System Overview**
The evaluation framework is built around the concept of structured assessment for AI-generated fables. The main idea is to establish a standardized approach where fables are evaluated based on key storytelling criteria: grammar, creativity, and alignment with predefined narrative structures. Additionally, the system measures the **diversity** of generated stories to ensure variability in themes, syntax, and character usage. By leveraging an AI model, the framework provides objective scoring and recommendations for improvement, making it a scalable solution for assessing large datasets of AI-generated creative writing.

## **3. Configuration and Evaluation Process**

### **Evaluation Prompts**
The system uses predefined prompts to guide evaluations. The **fable evaluation prompt** analyzes grammar, creativity, and consistency, returning structured JSON. The **diversity evaluation prompt** computes a diversity score (0-1) and identifies redundant patterns.

### **Structured Input Format**
```yaml
character: Fox
trait: Clever
setting: Forest
conflict: Helping a friend
resolution: Learns a lesson
moral: Kindness is rewarded
generated_fab: "Once there was a clever fox..."
```

## **4. Evaluation Framework**

To ensure consistency and objectivity, a specific model version (**GPT-4-Turbo-2024-01-25**) has been pinned. This guarantees that all evaluations are performed using the same AI model, preventing discrepancies that could arise from model updates. By using a fixed version, the framework ensures reproducibility in scoring, making comparisons across different fables and datasets more reliable.

### **4.1. `gpt_eval.py` - Core Evaluation**
This script calls **GPT-4-Turbo** to evaluate fables and assess dataset diversity.

#### **Fable Evaluation (`evaluate()`)**
```python
response = client.chat.completions.create(
    model="gpt-4-turbo-2024-01-25",
    messages=[
        {"role": "system", "content": "You are an evaluator of fables."},
        {"role": "user", "content": evaluation_prompt}
    ]
)
```
Returns structured JSON with grammar, creativity, consistency scores, and estimated age group.

#### **Diversity Evaluation (`evaluate_diversity()`)**
```python
response = client.chat.completions.create(
    model="gpt-4-turbo-2024-01-25",
    messages=[
        {"role": "system", "content": "Evaluate dataset diversity."},
        {"role": "user", "content": diversity_prompt}
    ]
)
```
Analyzes lexical, syntactic, and thematic variety, returning a **diversity score**.

## **5. CLI and Batch Processing**

### **5.1. `evals_cli.py` - Command Line Interface**
Extracts JSON results and ensures structured output. Example command:
```sh
python evals_cli.py --yaml_path evals_config.yml \
  --evaluation_type evaluation_prompt \
  --character Fox --trait Clever --setting Forest \
  --conflict "Helping a friend" --resolution "Learns a lesson" \
  --moral "Kindness is rewarded" --generated_fab "Once there was..."
```

### **5.2. `eval_model.py` - Batch Processing**
Loads datasets from CSV, updates YAML, and runs batch evaluations. Key functions:
```python
def update_yaml_with_fables(yaml_path, fables):
    with open(yaml_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    config["data"] = fables
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)
```
Runs evaluations and saves results in `evaluation_results.json`.

## **6. Output Formats**
### **Fable Evaluation Output**
```json
{
  "Grammar": "9/10",
  "Creativity": "8/10",
  "Consistency": "10/10",
  "Age group": "10-12",
  "Comments": "The fable is well-structured but could improve its resolution."
}
```

### **Diversity Evaluation Output**
```json
{
  "diversity_score": 0.82,
  "analysis": {
    "vocabulary_variation": 0.9,
    "themes_variation": 0.85
  },
  "suggested_improvements": ["Increase variety in settings and characters"]
}
```

## **7. Conclusion**
This system provides an automated, structured evaluation for AI-generated fables. It ensures consistency in storytelling while promoting diversity in AI-generated narratives. Future improvements may include **fine-tuned scoring models and expanded multilingual support**.

https://arxiv.org/pdf/2303.16634 https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
https://arxiv.org/pdf/2303.16634