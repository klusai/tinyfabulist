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
    Ești un expert în traduceri literare specializat în fabule românești. Sarcina ta este să îmbunătățești fabula tradusă oferită
    pe baza evaluărilor și feedback-ului specific.

    Fabula originală în engleză:
    {original_fable}

    Fabula tradusă în prezent în română:
    {translated_fable}

    Evaluări (scală de la 1 la 10):
    - Acuratețea traducerii: {ratings['translation_accuracy']}
    - Fluență: {ratings['fluency']}
    - Păstrarea stilului: {ratings['style_preservation']}
    - Claritatea moralei: {ratings['moral_clarity']}

    Feedback:
    - {explanations[0]} (Acuratețea traducerii)
    - {explanations[1]} (Fluență)
    - {explanations[2]} (Păstrarea stilului)
    - {explanations[3]} (Claritatea moralei)

    Îmbunătățește fabula tradusă abordând aceste probleme în timp ce menții naturalețea și coerența ei în limba română.
    Oferă DOAR traducerea îmbunătățită, fără comentarii sau explicații suplimentare.
    Asigură-te că traducerea este exclusiv în limba română!
    
    Nu repeta instrucțiunile sau prompt-ul. Răspunde doar cu textul tradus îmbunătățit.
    """

    # Chat-based translation (LLM approach)
    client = OpenAI(base_url=endpoint, api_key=api_key)

    system_prompt = "Ești un asistent de traducere. Tradu textul următor din limba engleză în limba română. Returnează doar textul tradus."
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


def extract_romanian_fable(text, original_translation):
    """
    Extract only the Romanian fable from the model's response, removing any instructions or meta-text.
    
    Parameters:0
        text: Text output from the model
        original_translation: Original translation to fall back on
        
    Returns:
        str: Clean Romanian fable text
    """
    # List of instruction text that needs to be removed
    instruction_texts = [
        "abordând aceste probleme în timp ce menții naturalețea și coerența ei în limba română",
        "Oferă DOAR traducerea îmbunătățită, fără comentarii sau explicații suplimentare",
        "Asigură-te că traducerea este exclusiv în limba română",
        "Nu repeta instrucțiunile sau prompt-ul",
        "Răspunde doar cu textul tradus îmbunătățit",
    ]
    
    # Remove instruction text blocks
    for instruction in instruction_texts:
        text = text.replace(instruction, "")
    
    # Remove the entire prompt if it got repeated in the response
    prompt_markers = [
        "Ești un expert în traduceri literare",
        "Fabula originală în engleză:",
        "Fabula tradusă în prezent în română:",
        "Evaluări (scală de la 1 la 10):",
        "Acuratețea traducerii:",
        "Feedback:",
        "Îmbunătățește fabula tradusă",
    ]
    
    # Find the position after all prompt parts
    last_marker_pos = -1
    for marker in prompt_markers:
        pos = text.find(marker)
        if pos > -1:
            marker_end = pos + len(marker)
            last_marker_pos = max(last_marker_pos, marker_end)
    
    # If we found any prompt parts, start after them
    if last_marker_pos > -1:
        text = text[last_marker_pos:].strip()
    
    # Remove common prefixes
    common_prefixes = [
        "Iată traducerea îmbunătățită:",
        "Traducerea îmbunătățită:",
        "Fabula îmbunătățită:",
        "Versiunea îmbunătățită:",
        "Iată fabula tradusă îmbunătățită:",
        "Textul îmbunătățit:",
        "Traducere:", 
        "Fabula:",
        "```"
    ]
    
    for prefix in common_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove common suffixes
    common_suffixes = [
        "```",
        "Aceasta este traducerea îmbunătățită.",
        "Aceasta este versiunea îmbunătățită."
    ]
    
    for suffix in common_suffixes:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
    
    # Clean up any excessive whitespace or newlines
    text = ' '.join(text.split())
    
    # If the text was butchered and nothing meaningful remains, return the original
    if len(text) < 50 and original_translation and len(original_translation) > 100:
        logger.warning("Extraction resulted in very short text, using original translation")
        return original_translation
    
    return text.strip()


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
        enhanced_entry["llm_name"] = "_Enhanced-Llama-3.3-70B"
        
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
    
    input_file = "/home/ap/Documents/Work/Research/tiny_fabulist/tinyfabulist/data/evaluations_ro/tf_fables_llama-3-1-8b-instruct-mpp_dt250310-094515_translation_ro_Llama-3.3-70B-Instruct_250318-093757_eval_e.jsonl_jsonl_eval_eo3-mini-2025-01-31_dt20250318_125017.jsonl"
    output_file = f"/home/ap/Documents/Work/Research/tiny_fabulist/tinyfabulist/data/translations/tf_enhanced_{timestamp}.jsonl"
    
    # Log the output filename
    logger.info(f"Output will be saved to: {output_file}")
    
    # Set the number of parallel workers (adjust based on your CPU and API rate limits)
    max_workers = 34
    
    enhance_jsonl(input_file, output_file, max_workers)