import argparse
import logging

from .pytorch_extender import PyTorchExtender


def setup_logging(log_filename: str) -> None:
    """Set up logging to file and stream handlers."""
    formatter = logging.Formatter(
        "%(levelname)s [%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
    )

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])


def extend_pytorch(args: argparse.Namespace) -> None:
    """Extend PyTorch input trace in json format to multi GPU version."""
    extender = PyTorchExtender()
    """testing."""
    filenames = []
    id = []
    for i in range(2):
        filenames.append(f"{args.input}{i}")
        id.append(i)
    extender.extend(id, filenames)


def main() -> None:
    """Extend chakra trace to multiple GPU version using data / tensor / model / pipeline para. With the cluster info and traces already known."""
    parser = argparse.ArgumentParser(
        description=(
            "Extend chakra trace to multiple GPU version using data / tensor / model / pipeline para. "
            "Current support tensor / model para."
            "And just need the traces already known."
            "The extender just take the prefix name of the already known traces', and extend the trace to multiple gpu."
            "The name should be alpha0.json, alpha1.json, alpha2.json etc. which --input = alpha. If you have cluster info(building), then some traces not mentioned in the info can be noting."
        )
    )

    parser.add_argument(
        "--log-filename", type=str, default="debug.log", help="Log filename"
    )

    subparsers = parser.add_subparsers(
        title="subcommands", description="Valid subcommands", help="Input type"
    )
    pytorch_extend_parser = subparsers.add_parser(
        "PyTorch",
        help="Extend Chakra host + device execution trace in JSON to multi GPUs"
        "still in JSON",
    )
    pytorch_extend_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Chakara host + device traces in the JSON format. If we call the input as `name`, so the traces should be $name$i.json, which $i = 0, 1, 2 etc. ",
    )

    args = parser.parse_args()

    if "func" in args:
        setup_logging(args.log_filename)
        args.func(args)
        logging.info(
            f"Conversion successful. Output file is available at {args.output}."
        )
    else:
        parser.print_help()
    pytorch_extend_parser.set_defaults(func=extend_pytorch)


if __name__ == "__main__":
    main()
