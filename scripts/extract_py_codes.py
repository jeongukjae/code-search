import glob
import ast
import zipfile
from dataclasses import dataclass, asdict
import os
import json

from absl import logging, flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_dir", None, "Input directory to save the codes.", required=True
)
flags.DEFINE_string(
    "output_dir", None, "Output directory to save the codes.", required=True
)


@dataclass
class Code:
    repo_name: str
    ref: str
    file: str
    line: int
    end_line: int
    code: str


def main(argv):
    zip_files = glob.glob(FLAGS.input_dir + "/*.zip")

    for zip_file in zip_files:
        logging.info(f"Extracting codes from {zip_file}...")

        repo_name, ref = os.path.basename(zip_file).removesuffix(".zip").rsplit("_", 1)
        codes = []
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for file in zip_ref.namelist():
                if not file.endswith(".py"):
                    continue

                with zip_ref.open(file) as f:
                    filepath = os.path.sep.join(file.split(os.path.sep)[1:])
                    codes.extend(
                        extract_py_codes(
                            repo_name,
                            ref,
                            filepath,
                            f.read().decode("utf-8"),
                        )
                    )

        basename = os.path.basename(zip_file)
        basename = basename.replace(".zip", ".jsonl")
        output_uri = os.path.join(FLAGS.output_dir, basename)
        with open(output_uri, "w") as f:
            for code in codes:
                f.write(json.dumps(asdict(code)) + "\n")


def extract_py_codes(repo_name, ref, file, content):
    """Extract all functions from a python source code."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    codes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            codes.append(
                Code(
                    repo_name=repo_name,
                    ref=ref,
                    file=file,
                    line=node.lineno,
                    end_line=node.end_lineno,
                    code=ast.get_source_segment(content, node),
                )
            )
    return codes


if __name__ == "__main__":
    app.run(main)
