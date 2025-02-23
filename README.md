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

Generate 10 random fable prompts in text format:

```bash
python tinyfabulist.py --generate-prompts --count 10 > prompts.jsonl
```

Generate fables from a JSONL file using all available models:

```bash
python tinyfabulist.py --generate-fables prompts.jsonl
```

Generate fables from a JSONL file using specific models:

