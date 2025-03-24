from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.deepl import deepl_subparser
from tinyfabulist.translate.enhance import enhnace_entry_point
from tinyfabulist.translate.gpt import gpt_subparser
from tinyfabulist.translate.open_source import open_source_translate_subparser

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

def get_translate_subparser(engine):
    if engine == "deepl":
        return get_deepl_subparser()
    elif engine == "gpt":
        return get_gpt_subparser()
    else:
        return get_open_source_subparser()

def enhance_translation(args):
    logger.info("Enhancing translation")
    return enhnace_entry_point(args.input, args.input_yaml)