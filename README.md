# TinyFabulist
Set of LLM-based tools for generating millions of unique fables by combining storytelling patterns, linguistic creativity, and moral lessons. Designed for language modeling, NLP research, and educational applications.

## Overview

TinyFabulist is a Python-based fable generation system that:

1. Uses a rich YAML configuration to define story elements:
   - Over 100 animal characters from diverse habitats
   - 50+ personality traits and characteristics 
   - 50+ unique settings and locations
   - Extensive lists of conflicts, resolutions and morals
   - Customizable prompt templates

2. Provides flexible generation options:
   - Random or sequential story element selection
   - Configurable output formats (text, CSV, JSONL)
   - Support for multiple LLM models via HuggingFace endpoints
   - Batch generation capabilities

3. Generates structured fables with:
   - Vivid scene-setting and descriptions
   - Natural character development
   - Meaningful dialogue
   - Clear moral lessons
   - Appropriate length (~250 words)

4. Features robust error handling:
   - Configuration validation
   - Logging system
   - Graceful error recovery
   - Detailed error messages

5. Supports educational and research use cases:
   - Language modeling experiments
   - NLP dataset generation
   - Creative writing assistance
   - Moral education resources

## Usage Examples
Below is a usage guide for the `tinyfabulist.py` file, which serves as the main entry point for both generating and evaluating fables. This file uses subcommands to separate functionality.

---

Run the script with one of the two subcommands: **generate** or **evaluate**.

```bash
python tinyfabulist.py <command> [options]
```

---

## Subcommands and Their Arguments

### 1. **generate**

This subcommand is used to either generate fable prompts or generate complete fables based on a JSONL prompt file.

#### Arguments for `generate`:

- `--generate-prompts`  
  *Description:* Generate fable prompts only.  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-prompts --count 10 --output jsonl > prompts.jsonl
  ```

- `--generate-fables <file>`  
  *Type:* `str`  
  *Description:* Generate complete fables from the provided JSONL prompt file.  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-fables prompts.jsonl --output text
  ```

- `--randomize`  
  *Description:* Randomize the selection of story elements (characters, traits, settings, etc.).  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-prompts --randomize --count 20
  ```

- `--output <format>`  
  *Choices:* `text`, `jsonl`, `csv`  
  *Default:* `text`  
  *Description:* Set the output format for generated prompts or fables.  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-fables prompts.jsonl --output csv
  ```

- `--output-file <file>`  
  *Type:* `str`  
  *Default:* `results.jsonl`  
  *Description:* This file is used to load existing hashes to avoid duplicate fable generation.  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-fables prompts.jsonl --output jsonl --output-file my_fables.jsonl
  ```

- `--count <number>`  
  *Type:* `int`  
  *Default:* `100`  
  *Description:* Number of fable prompts to generate (used only when `--generate-prompts` is specified).  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-prompts --count 50
  ```

- `--models <model_names>`  
  *Type:* List of strings (`nargs='+'`)  
  *Description:* Specify which LLM models (as defined in your configuration) to use for generating fables. If not provided, all available models in the configuration are used.  
  *Example:*  
  ```bash
  python tinyfabulist.py generate --generate-fables prompts.jsonl --models model1 model2
  ```

---

### 2. **evaluate**

This subcommand is used to evaluate already generated fables from a JSONL file.

#### Arguments for `evaluate`:

- `--evaluate <file>`  
  *Type:* `str`  
  *Required:* Yes  
  *Description:* Path to the JSONL file containing the fables to evaluate.  
  *Example:*  
  ```bash
  python tinyfabulist.py evaluate --evaluate results.jsonl
  ```

---

## Full Examples

1. **Generating Fable Prompts in JSONL Format:**

   ```bash
   python tinyfabulist.py generate --generate-prompts --count 10 --output jsonl > prompts.jsonl
   ```

2. **Generating Fables from a Prompt File with Deduplication (Using Specific Models):**

   ```bash
   python tinyfabulist.py generate --generate-fables prompts.jsonl --output jsonl --models model1 model2
   ```

3. **Evaluating Generated Fables:**

   ```bash
   python ./tinyfabulist.py evaluate --jsonl data/fables/
   ```

4. **Presenting Stats:**

   ```bash
   python ./tinyfabulist.py stats --jsonl data/evaluations
   ```

5. **Translate module Stats:**

   ```bash
   python tinyfabulist.py translate --input <file.jsonl> --target-lang RO
   ```
