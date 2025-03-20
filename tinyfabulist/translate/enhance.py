import json
import time
from openai import OpenAI
import requests
import concurrent.futures
from dotenv import load_dotenv
from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.utils import read_api_key, load_translator_config

logger = setup_logging()

input_yaml = "/home/ap/Documents/Work/Research/tiny_fabulist/tinyfabulist.yaml"


def improve_translation(original_fable, translated_fable, ratings, explanations, api_key, endpoint):
    """
    Uses an open-source translation model to refine the translated fable based on ratings and feedback.
    """
    prompt = f"""
    As an expert Romanian literary translator, analyze the following translation:

    Original English text:
    {original_fable}

    Current Romanian translation:
    {translated_fable}

    Ratings (1-10):
    - Accuracy: {ratings['translation_accuracy']}
    - Fluency: {ratings['fluency']} 
    - Style: {ratings['style_preservation']}
    - Moral clarity: {ratings['moral_clarity']}

    Feedback:
    - Accuracy: {explanations[0]}
    - Fluency: {explanations[1]}
    - Style: {explanations[2]}
    - Moral clarity: {explanations[3]}

    Instructions:
    1. First identify all grammatical errors, gender/pronoun issues, mistranslations, and stylistic problems.
    2. Based on this analysis, provide ONLY the fully revised Romanian translation, preserving the style and moral of the original.

    Do not include any explanations, analysis, or other text - return exclusively the final corrected translation, in Romanian.
    """

    # Chat-based translation (LLM approach)
    client = OpenAI(base_url=endpoint, api_key=api_key)

    system_prompt = "You are an expert in literary translation, specialized in Romanian fables, with meticulous attention to linguistic, cultural, and stylistic details. Your task is to revise and enhance the Romanian translation of the fable by integrating the ratings and feedback provided."
    fable_prompt = prompt

    for attempt in range(3):
        try:
            chat_completion = client.chat.completions.create(
                model="tgi",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fable_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
                stream=True,
            )

            fable_translation = ""
            for message in chat_completion:
                if message.choices[0].delta.content is not None:
                    fable_translation += message.choices[0].delta.content
            return fable_translation
        except Exception as e:
            logger.error(f"Error on attempt {attempt+1}: {e}. Retrying...")
            time.sleep(5)

    return translated_fable # Return original translation if all attempts fail


def process_entry(entry_data):
    """
    Process a single entry for multithreading.
    
    Parameters:
        entry_data: Tuple containing (index, entry, api_key, endpoint)
        
    Returns:
        Tuple of (index, enhanced_entry or original_entry, success_flag)
    """
    i, entry, api_key, endpoint = entry_data
    
    try:
        # Extract fields based on the actual JSON structure
        original_fable = entry.get("fable", "")
        translated_fable = entry.get("translated_fable", "")
        
        # Extract evaluation
        evaluation = entry.get("evaluation", {})
        
        # Only proceed if we have both a fable and its translation
        if not original_fable or not translated_fable:
            logger.warning(f"Skipping entry {i+1}: Missing fable or translation")
            return i, entry, False
        
        # Extract ratings using the correct field names
        ratings = {
            "translation_accuracy": evaluation.get("translation_accuracy", 5),
            "fluency": evaluation.get("fluency", 5),
            "style_preservation": evaluation.get("style_preservation", 5),
            "moral_clarity": evaluation.get("moral_clarity", 5),
        }
        
        # Extract explanations from the explanation array
        explanations = evaluation.get("explanation", [""] * 4)
        # Ensure we have 4 explanation items
        while len(explanations) < 4:
            explanations.append("")
        
        # Improve translation
        improved_translation = improve_translation(
            original_fable, translated_fable, ratings, explanations, api_key, endpoint
        )
        
        # Create a new entry with the improved translation
        enhanced_entry = entry.copy()
        # Store the improved translation in the "translated_fable" field
        enhanced_entry["translated_fable"] = improved_translation
        enhanced_entry["llm_name"] = "_Enhanced-Llama-3.3-70B_Fine_Prompted"
        
        return i, enhanced_entry, True
    except Exception as e:
        logger.error(f"Error enhancing entry {i+1}: {e}")
        return i, entry, False


def enhance_jsonl(input_file, output_file, max_workers=34):
    """
    Reads a JSONL file, improves the translated fables using multithreading, and writes the enhanced data to a new JSONL file.
    
    Parameters:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        max_workers: Maximum number of parallel workers (default: 8)
    """
    start_time = time.time()
    load_dotenv()

    api_key = read_api_key("HF_ACCESS_TOKEN")
    # Fix: Use the correct path for the config file
    config = load_translator_config(input_yaml, "translator_ro") 
    endpoint = config.get("endpoint")

    if not endpoint:
        logger.critical("No endpoint found for translation model.")
        raise ValueError("Translation model endpoint not found.")

    # Read JSONL file entries
    entries = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line: {e}")
                continue
    
    logger.info(f"Found {len(entries)} entries to process")
    
    # Prepare data for multithreading
    entry_data = [(i, entry, api_key, endpoint) for i, entry in enumerate(entries)]
    
    # Process entries in parallel
    enhanced_entries = [None] * len(entries)  # Pre-allocate list to maintain order
    success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entry, data): data[0] for data in entry_data}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            original_index, result_entry, success = future.result()
            enhanced_entries[original_index] = result_entry
            
            if success:
                success_count += 1
            
            # Log progress periodically
            if (i + 1) % 5 == 0 or i + 1 == len(entries):
                logger.info(f"Progress: {i + 1}/{len(entries)} entries processed ({(i + 1) / len(entries) * 100:.1f}%)")
    
    # Write enhanced entries to output JSONL file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in enhanced_entries:
            if entry:  # Skip None entries if any
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")

    elapsed_time = time.time() - start_time
    logger.info(f"Enhanced {success_count} out of {len(entries)} fables in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {output_file}")


# Run the script on a JSONL file
if __name__ == "__main__":
    from datetime import datetime
    
    # Generate timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input_file = "tinyfabulist/data/evaluations_ro/evaluations_eval_e_gpt_20250319_121834_tf_fables_llama-3-1-8b-instruct-mpp_dt250310-094515_translation_ro_Llama-3.3-70B-Instruct_250318-093757_eval_e.jsonl"
    output_file = f"tinyfabulist/data/translations/tf_enhanced_{timestamp}.jsonl"
    
    # Log the output filename
    logger.info(f"Output will be saved to: {output_file}")
    
    # Set the number of parallel workers (adjust based on your CPU and API rate limits)
    max_workers = 34
    
    enhance_jsonl(input_file, output_file, max_workers)