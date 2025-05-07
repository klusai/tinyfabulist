def add_translate_subparser(
    subparsers, function, source_lang="eng_Latn", target_lang="ron_Latn"
) -> None:
    """
    Add the translate subparser to the main parser.
    """
    translate_parser = subparsers.add_parser(
        "translate", help="Translate fables from a JSONL file"
    )

    translate_parser.add_argument(
        "--input", required=True, help="Path to input JSONL file containing fables"
    )

    translate_parser.add_argument(
        "--output",
        choices=["text", "jsonl", "csv"],
        default="text",
        help="Output format (default: text)",
    )

    translate_parser.add_argument(
        "--source-lang", default=source_lang, help="Source language code"
    )

    translate_parser.add_argument(
        "--target-lang", default=target_lang, help="Target language code"
    )

    translate_parser.add_argument(
        "--models", nargs="+", help="Specify models to use (as defined in configuration)"
    )

    translate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=500,
        help="Maximum number of concurrent requests (default: 500)",
    )

    translate_parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars for batch processing",
    )

    translate_parser.set_defaults(func=function)
