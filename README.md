# TinyFabulist

**Set of LLM-based tools for generating millions of unique fables by combining storytelling patterns, linguistic creativity, and moral lessons.** Designed for language modeling, NLP research, and educational applications.

---

## Overview

TinyFabulist is a Python-based fable generation system that:

- **Uses a rich YAML configuration to define story elements:**
  - Over 100 animal characters from diverse habitats
  - 50+ personality traits and characteristics
  - 50+ unique settings and locations
  - Extensive lists of conflicts, resolutions, and morals
  - Customizable prompt templates

- **Provides flexible generation options:**
  - Random or sequential story element selection
  - Configurable output formats (text, CSV, JSONL)
  - Support for multiple LLM models via HuggingFace endpoints
  - Batch generation capabilities

- **Generates structured fables with:**
  - Vivid scene-setting and descriptions
  - Natural character development
  - Meaningful dialogue
  - Clear moral lessons
  - Appropriate length (~250 words)

- **Features robust error handling:**
  - Configuration validation
  - Logging system
  - Graceful error recovery
  - Detailed error messages

- **Supports educational and research use cases:**
  - Language modeling experiments
  - NLP dataset generation
  - Creative writing assistance
  - Moral education resources

---

## Usage Examples

The `tinyfabulist.py` file serves as the main entry point for both generating and evaluating fables. Use the provided subcommands to separate functionality.

Run the script with one of the two subcommands: **generate** or **evaluate**.

```bash
python tinyfabulist.py <command> [options]
```

---

## Subcommands

### 1. Generate

Generate either prompts or complete fables from JSONL prompt files.

**Arguments:**

- `--generate-prompts`  
  Generate only fable prompts.

- `--generate-fables <file>`  
  Generate fables from the specified JSONL prompt file.

- `--randomize`  
  Randomize selection of story elements.

- `--output <format>`  
  Output format: text (default), jsonl, or csv.

- `--input-file <file>`  
  Specify input file for deduplication or reference.

- `--count <number>`  
  Number of prompts to generate (default: 100).

- `--models <model_names>`  
  Specify which models to use (as defined in configuration).

**Examples:**

```bash
python tinyfabulist.py generate --generate-prompts --count 10 --output jsonl > prompts.jsonl
```

```bash
python tinyfabulist.py generate --generate-fables prompts.jsonl --output jsonl --models model1 model2
```

---

### 2. Evaluate

Evaluate generated fables from JSONL files or directories.

**Arguments:**

- `--input <file_or_directory>` (required)  
  JSONL file or directory containing files starting with `tf_fables`.

**Example:**

```bash
python ./tinyfabulist.py evaluate --jsonl data/fables/
```

---

### 3. Translate

Translate JSONL content (fables/prompts) into Romanian.

**Arguments:**

- `--input` (required)  
  Path to the input JSONL file.

- `--outputPath`  
  Output translated file (default: input_filename_ro.jsonl).

- `--configYAML`  
  Configuration file path (default: tinyfabulist.yaml).

- `--translator-key`  
  Translator configuration key in YAML file (default: translator_ro).

- `--source-lang`  
  Source language code (default: eng_Latn).

- `--target-lang`  
  Target language code (default: ron_Latn).

- `--batch-size`  
  Records processed per batch before saving (default: 100).

- `--fields`  
  Comma-separated list of fields to translate (default: fable,prompt).

- `--max-workers`  
  Maximum number of threads for parallel processing (default: 200).

**Example:**

```bash
python tinyfabulist.py translate --input <file.jsonl> --target-lang RO
```

---

### 4. Stats

Compute and display aggregated statistics from evaluation JSONL files.

**Arguments:**

- `--inputPath`  
  Path to JSONL file or directory (default: evaluate.jsonl).

- `--output-mode`  
  Output location: terminal, files, or both (default: both).

- `--plot-mode`  
  Plotting library: plotly or matplotlib (default: plotly).

**Example:**

```bash
python ./tinyfabulist.py stats --jsonl data/evaluations
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

5. **Translate Module:**

   ```bash
   python tinyfabulist.py translate --input <file.jsonl> --target-lang RO
   ```
