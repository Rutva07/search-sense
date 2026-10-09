import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SearchSense reproducible pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--shards", nargs="+", type=int, default=[2])
    p.add_argument("--output", type=Path, default=Path("data/raw"))
    p = sub.add_parser("ingest")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--database", type=Path, default=Path("data/events.sqlite"))
    p.add_argument("--max-rows", type=int)
    p = sub.add_parser("train")
    p.add_argument("--database", type=Path, default=Path("data/events.sqlite"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--max-groups", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("evaluate")
    p.add_argument("--database", type=Path, default=Path("data/events.sqlite"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "download":
        from .data import download
        result = [str(p) for p in download(**args)]
    elif command == "ingest":
        from .data import ingest
        result = ingest(**args)
    elif command == "train":
        from .pipeline import train
        args["directory"] = args.pop("artifacts")
        result = train(**args)
    elif command == "evaluate":
        from .pipeline import evaluate
        args["directory"] = args.pop("artifacts")
        result = evaluate(**args)
    else:
        import uvicorn
        uvicorn.run("searchsense.api:app", **args, workers=1)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
