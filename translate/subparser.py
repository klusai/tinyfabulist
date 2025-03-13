def add_translate_subparser(subparsers, function, source_lang='eng_Latn', target_lang='ron_Latn') -> None:
    """
    Add the translate subparser to the main parser.
    """
    translate_parser = subparsers.add_parser(
        'translate', 
        help='Translate content in a JSONL file to Romanian'
    )
    
    translate_parser.add_argument(
        '--input', 
        required=True,
        help='Path to input JSONL file'
    )
    
    translate_parser.add_argument(
        '--output', 
        help='Path to output translated JSONL file (default: input_filename_ro.jsonl)'
    )
    
    translate_parser.add_argument(
        '--config', 
        default='tinyfabulist.yaml',
        help='Path to YAML configuration file (default: tinyfabulist.yaml)'
    )
    
    translate_parser.add_argument(
        '--translator-key', 
        default='translator_ro',
        help='Key in the YAML config file for the Romanian translator (default: translator_ro)'
    )
    
    translate_parser.add_argument(
        '--source-lang', 
        default=source_lang,
        help='Source language code'
    )
    
    translate_parser.add_argument(
        '--target-lang', 
        default=target_lang,
        help='Target language code'
    )
    
    translate_parser.add_argument(
        '--batch-size', 
        type=int,
        default=100,
        help='Number of records to process before saving progress (default: 100)'
    )
    
    translate_parser.add_argument(
        '--fields', 
        help='Comma-separated list of fields to translate (default: fable,prompt)'
    )
    
    translate_parser.add_argument(
        '--max-workers', 
        type=int,
        default=30,
        help='Maximum number of threads to use (default: 30)'
    )
    
    translate_parser.set_defaults(func=function)