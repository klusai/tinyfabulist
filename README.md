# TinyFabulist

**Set of LLM-based tools for generating millions of unique fables by combining storytelling patterns, linguistic creativity, and moral lessons.** Designed for language modeling, NLP research, and educational applications.

---

This repository contains the **code used to generate the <a href="https://huggingface.co/datasets/klusai/ds-tf1-en-3m">ds-tf1-en-3m</a> dataset**, which is available on Hugging Face. The dataset includes 3 million fables in English, intended for training and fine-tuning natural language processing (NLP) models.

---

## Docker Usage

TinyFabulist can be run in a Docker container, providing a consistent environment across different systems.

### Building the Docker Image

From the project root directory, build the Docker image:

```bash
# Build the image with tag 'tiny_fabulist'
docker build -t tiny_fabulist .
```

### Running the Docker Image

#### Basic usage (shows help)
```bash
docker run tiny_fabulist
```

#### Interactive shell access (recommended)
```bash
docker run -it --entrypoint /bin/bash tiny_fabulist
```

#### Running with specific commands
```bash
docker run tiny_fabulist generate --generate-prompts --count 10
```

#### Mounting data directory for persistence
```bash
docker run -v $(pwd)/data:/app/data tiny_fabulist [COMMAND]
```

# Example: Generate fables and save to mounted volume
```bash
docker run -v $(pwd)/data:/app/data tiny_fabulist generate --generate-fables data/prompts.jsonl --output jsonl
```

### Using Docker Compose (Recommended)

TinyFabulist provides a Docker Compose configuration for easier usage and development. This is the recommended way to run TinyFabulist as it handles all the volume mounting and environment setup automatically.

#### Prerequisites

Make sure you have Docker and Docker Compose installed:

```bash
# Check Docker installation
docker --version

# Check Docker Compose installation
docker compose version
```

If you need to install Docker Compose, follow the [official installation instructions](https://docs.docker.com/compose/install/).

#### Standard Usage

Running TinyFabulist with Docker Compose:

```bash
# Show help information
docker compose run --rm app

# Generate prompts (10 prompts in JSONL format)
docker compose run --rm app generate --generate-prompts --count 10

# Generate fables from a prompts file
docker compose run --rm app generate --generate-fables data/prompts.jsonl --output jsonl

# Run with any other command
docker compose run --rm app [COMMAND]
```

The `app` service automatically mounts your local `data/` directory to the container's `/app/data`, ensuring that all generated files are persisted on your host machine.

#### Development with Hot Reloading

For development, TinyFabulist provides a special service with hot reloading capabilities:

```bash
# Start the development environment
docker compose run --rm dev
```

This opens an interactive shell where:
- TinyFabulist is installed in editable mode (`pip install -e .`)
- Python files are watched for changes and code is automatically reloaded
- You can run any TinyFabulist command manually
- Changes to your code are immediately available without container restarts

Inside the development environment, you can run commands like:

```bash
# Generate 5 prompts
python tinyfabulist.py generate --generate-prompts --count 5

# Generate fables
python tinyfabulist.py generate --generate-fables data/prompts.jsonl
```

When you modify any Python file in the project, you'll see a notification that the code has been reloaded. Then you can simply run your command again to use the updated code.

#### Docker Compose Benefits

1. **Persistent Data:** Changes to the `data/` directory are saved to your host machine
2. **Simplified Commands:** No need to remember complex docker run arguments
3. **Development Mode:** Edit code on your host while running in the container
4. **Consistent Environment:** Same environment across all development machines

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

- **Provides comprehensive translation capabilities:**
  - Support for multiple translation engines (GPT, DeepL, open-source models)
  - Enhanced translations with quality improvements
  - Parallel processing for efficient batch translation

---

## Usage Examples
Below is a usage guide for the `tinyfabulist.py` file, which serves as the main entry point for both generating and evaluating fables. This file uses subcommands to separate functionality.

The `tinyfabulist.py` file serves as the main entry point for generating, evaluating, translating, and enhancing fables. Use the provided subcommands to separate functionality.

```bash
python tinyfabulist.py <command> [options]
```

Available commands:
- `generate`: Generate fable prompts or fables
- `evaluate`: Evaluate generated fables
- `stats`: Compute statistics from evaluation files
- `translate`: Translate fables to other languages
- `enhance`: Enhance existing translations with refinements

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

TinyFabulist provides evaluation capabilities for both original English fables and their Romanian translations.

#### English Evaluation

Evaluate generated English fables for quality, creativity, moral clarity, and adherence to prompt.

**Arguments:**

- `--input <file_or_directory>` (required)  
  JSONL file or directory containing files starting with `tf_fables`.

**Example:**

```bash
python tinyfabulist.py evaluate --input tinyfabulist/data/fables/
```

#### Romanian Evaluation

Evaluate translated Romanian fables for translation accuracy, fluency, style preservation, and moral clarity.

**Arguments:**

- `--input <file_or_directory>` (required)  
  JSONL file or directory containing translated fables.

**Example:**

```bash
python tinyfabulist.py evaluate_ro --input tinyfabulist/data/translations/
```

The evaluation results are stored in JSONL format with these key metrics:
- For English: grammar, creativity, moral_clarity, adherence_to_prompt
- For Romanian: translation_accuracy, fluency, style_preservation, moral_clarity

Each metric is scored on a scale of 1-10 with detailed explanations provided.

---

### 3. Translate

Translate fables to other languages with support for multiple translation engines.

**Arguments:**

- `--input <file>` (required)  
  Path to the input JSONL file containing fables to translate.

- `--engine {gpt,deepl,open_source,mbart}`  
  Select translation engine to use (default: gpt).

- `--output <file>`  
  Path to save the translated output (default: auto-generated with timestamp).

- `--config <file>`  
  Path to translation configuration file (default: conf/translator.yaml).

- `--translator-key <key>`  
  Key in the config file for the translator settings (default: translator_ro).

- `--batch-size <number>`  
  Number of items to process in a batch (default: 10).

- `--max-workers <number>`  
  Maximum number of worker threads (default: 5).

- `--fields <comma-separated-list>`  
  Comma-separated list of fields to translate (default: fable,prompt).

- `--source-lang <code>`  
  Source language code (overrides default).

- `--target-lang <code>`  
  Target language code (overrides default).

**Examples:**

```bash
# Translate using GPT (default)
python tinyfabulist.py translate --input path/to/fables.jsonl

# Translate using DeepL
python tinyfabulist.py translate --engine deepl --input path/to/fables.jsonl

# Translate using open-source model
python tinyfabulist.py translate --engine open_source --input path/to/fables.jsonl

# Translate using mBART
python tinyfabulist.py translate --engine mbart --input path/to/fables.jsonl
```

---

### 4. Enhance

Enhance existing translations with refinements based on evaluation feedback.

**Arguments:**

- `--input <file>` (required)  
  Path to the input JSONL file containing translations to enhance.

- `--input-yaml <file>`  
  Path to enhancement configuration YAML file (default: conf/enhance.yaml).

- `--max-workers <number>`  
  Maximum number of worker threads for parallel processing (default: 34).

- `--output <file>`  
  Path to save enhanced output (default: auto-generated with timestamp).

**Example:**

```bash
# Enhance translations with default settings
python tinyfabulist.py enhance --input path/to/translations.jsonl

# Enhance with custom configuration and output path
python tinyfabulist.py enhance --input path/to/translations.jsonl --input-yaml path/to/config.yaml --output enhanced_output.jsonl
```

---

### 5. Stats

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
python tinyfabulist.py stats --input data/evaluations
```

---

### 6. Visualize

Launch an interactive web dashboard for exploring and visualizing JSONL files.

**Arguments:**

- `--input <file>` (required)  
  Path to the JSONL file to visualize.

- `--port <number>`  
  Port to run the visualization server on (default: 8050).

- `--host <address>`  
  Host address to bind the server to (default: 127.0.0.1).

- `--debug`  
  Run the visualization server in debug mode, enabling auto-reload on code changes.

- `--no-browser`  
  Don't automatically open the browser when starting the server.

**Example:**

```bash
# Launch visualization with default settings
python tinyfabulist.py visualize --input path/to/fables.jsonl

# Launch visualization on a specific port
python tinyfabulist.py visualize --input path/to/evaluations.jsonl --port 8888

# Run on a public IP to allow remote access
python tinyfabulist.py visualize --input path/to/translations.jsonl --host 0.0.0.0
```

---

## Full Examples

1. **Generate Fable Prompts in JSONL Format:**

   ```bash
   python tinyfabulist.py generate --generate-prompts --count 10 --output jsonl
   ```

2. **Generate Fables from Prompts Using Specific Models:**

   ```bash
   python tinyfabulist.py generate --generate-fables prompts.jsonl --output jsonl --models model1 model2
   ```

3. **Translate Fables Using Different Engines:**

   ```bash
   # Using GPT
   python tinyfabulist.py translate --input fables.jsonl --engine gpt
   
   # Using DeepL with custom output
   python tinyfabulist.py translate --input fables.jsonl --engine deepl --output translated_deepl.jsonl
   ```

4. **Enhance Existing Translations:**

   ```bash
   python tinyfabulist.py enhance --input translated_fables.jsonl
   ```

5. **Evaluate Generated Fables:**

   ```bash
   python tinyfabulist.py evaluate --input tinyfabulist/data/fables/
   ```

6. **Generate Statistics from Evaluations:**

   ```bash
   python tinyfabulist.py stats --input tinyfabulist/data/evaluations
   ```

# Cute Raport

Welcome to the cutest raport ever on the generation datetime analysis! :sparkles:

## Overview

In this analysis, we processed the data from a JSONL file to extract the maximum and minimum generation datetimes. Using the power of JSON queries and unix commands, we computed the difference between the greatest and smallest timestamps.

## Detailed Analysis

- **Maximum Generation Datetime:** Obtained by scanning through all entries.
- **Minimum Generation Datetime:** Derived from the same dataset.
- **Difference:** The computed difference is **866 seconds** (roughly 14 minutes and 26 seconds).

## Machines and Execution

This entire analysis was executed using multiple machines in our Hugging Face endpoints environment. We leveraged the robust compute capabilities and seamless integration of the HF endpoints to parallelize and accelerate our data processing.

- **Number of Machines:** Our system is powered by a robust, multi-machine cluster in the Hugging Face endpoints environment. In our latest production run, this extraordinary setup generated a staggering **6,000** unique fables, clearly demonstrating the unparalleled scalability, efficiency, and high availability of our computational framework. Every machine played a critical role in achieving this remarkable output.
- **Key Platforms:** Everything is done in HF endpoints, ensuring scalability, integration, and smooth orchestration of the computational tasks.

## Conclusion

This raport demonstrates not only our capabilities in processing data efficiently across a number of machines using state-of-the-art HF endpoints, but also our commitment to generating positive outcomes in a friendly and approachable style.

**Scaling Impact:** In our experiments, using 20 AWS Inferentia machines, we executed 6,000 prompts in just 866 seconds. Extrapolating this performance, generating 3 million fables would take roughly 120 hours (about 5 days) at an estimated cost of around $2100. In contrast, when using A10G machines, a single unit would need approximately 660 hours (or 27.5 days) to handle the workload. However, with our breakthrough project Hydra and a robust 15-machine cluster, we can dramatically cut down the compute time to under 1 day. This transformation not only underscores the incredible benefits of scaling but also exemplifies our commitment to efficiency and cost-effectiveness in large-scale language model operations.

Stay tuned for more cute insights and further analyses!

:heart: :sparkles:
