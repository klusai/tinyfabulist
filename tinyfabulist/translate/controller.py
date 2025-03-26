from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.deepl import deepl_subparser
from tinyfabulist.translate.enhance import enhnace_entry_point
from tinyfabulist.translate.gpt import gpt_subparser
from tinyfabulist.translate.open_source import open_source_translate_subparser
from tinyfabulist.translate.mbart import mbart_translate_subparser

logger = setup_logging()

def get_deepl_subparser():
    logger.info("Running DeepL Translator")
    return deepl_subparser

def get_gpt_subparser():
    logger.info("Running GPT Translator")
    return gpt_subparser

def get_open_source_subparser():
    logger.info("Running Open-Source Translator")
    return open_source_translate_subparser

def get_mbart_subparser():
    logger.info("Running mBART Translator")
    return mbart_translate_subparser

def get_translate_subparser(engine):
    if engine == "deepl":
        return get_deepl_subparser()
    elif engine == "gpt":
        return get_gpt_subparser()
    elif engine == "mbart":
        return get_mbart_subparser()
    else:
        return get_open_source_subparser()

def enhance_translation(args):
    logger.info("Enhancing translation")
    return enhnace_entry_point(args.input, args.input_yaml)


def add_translate_subparser(subparsers):
    """Add the translate subparser with an engine option"""
    translate_parser = subparsers.add_parser("translate", help="Translate fables to other languages")
    
    # Add engine selection argument
    translate_parser.add_argument(
        "--engine",
        choices=["gpt", "deepl", "open_source", "mbart"],
        default="gpt",
        help="Select translation engine to use (default: gpt)"
    )
    
    # Add common translation arguments
    translate_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file containing fables to translate"
    )
    
    translate_parser.add_argument(
        "--output",
        type=str,
        help="Path to save the translated output (default: auto-generated with timestamp)"
    )
    
    translate_parser.add_argument(
        "--config",
        type=str,
        default="conf/translator.yaml",
        help="Path to translation configuration file"
    )

    # Add additional arguments that might be needed by different engines
    translate_parser.add_argument(
        "--translator-key",
        type=str,
        default="translator_ro",
        help="Key in the config file for the translator settings"
    )
    
    translate_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of items to process in a batch"
    )
    
    translate_parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of worker threads"
    )
    
    translate_parser.add_argument(
        "--fields",
        type=str,
        default="fable,prompt",
        help="Comma-separated list of fields to translate"
    )
    
    translate_parser.add_argument(
        "--source-lang",
        type=str,
        help="Source language code (overrides default)"
    )
    
    translate_parser.add_argument(
        "--target-lang",
        type=str,
        help="Target language code (overrides default)"
    )
    
    # Set the function that will handle the translate command
    translate_parser.set_defaults(func=handle_translate)


def handle_translate(args):
    """Handle the translate command by dispatching to the appropriate engine"""
    # Get the function for the selected engine
    engine_func = get_translate_subparser(args.engine)
    
    # Some engines expect args.source_lang and args.target_lang to be set
    if not hasattr(args, "source_lang") or not args.source_lang:
        if args.engine == "mbart":
            args.source_lang = "en_XX"
        elif args.engine == "gpt":
            args.source_lang = "en"
        elif args.engine == "deepl":
            args.source_lang = "EN"
        else:  # open_source
            args.source_lang = "EN"
            
    if not hasattr(args, "target_lang") or not args.target_lang:
        if args.engine == "mbart":
            args.target_lang = "ro_RO"
        elif args.engine == "gpt":
            args.target_lang = "ro"
        elif args.engine == "deepl":
            args.target_lang = "RO"
        else:  # open_source
            args.target_lang = "RO"
    
    # Call the engine function with the arguments
    engine_func(args)
