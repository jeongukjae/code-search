import torch
import numpy as np
import glob
import json

from absl import app, flags, logging
from tqdm import tqdm

from unixcoder import UniXcoder

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Input directory", required=True)
flags.DEFINE_string("output_dir", None, "Output directory", required=True)
flags.DEFINE_string("model_name", "microsoft/unixcoder-base", "Model name")
flags.DEFINE_integer("max_length", 512, "Max length of the input sequence")


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(FLAGS.model_name)
    model.to(device)

    files = glob.glob(FLAGS.input_dir + "/*.jsonl")
    logging.info(f"Found {len(files)} files")

    embeddings = []
    meta_infos = []

    for file in tqdm(files):
        with open(file) as f:
            for sample in f.readlines():
                sample = json.loads(sample)
                code = sample["code"]
                tokens_ids = model.tokenize(
                    [code], max_length=FLAGS.max_length, mode="<encoder-only>"
                )

                source_ids = torch.tensor(tokens_ids).to(device)
                with torch.no_grad():
                    _, code_embeddings = model(source_ids)
                    code_embeddings = torch.nn.functional.normalize(
                        code_embeddings, p=2, dim=1
                    )

                embeddings.append(code_embeddings.cpu().detach().numpy())
                meta_infos.append(
                    {
                        "repo_name": sample["repo_name"],
                        "ref": sample["ref"],
                        "file": sample["file"],
                        "line": sample["line"],
                        "end_line": sample["end_line"],
                    }
                )

    # save numpy and json
    embeddings = np.concatenate(embeddings, axis=0)

    np.save(FLAGS.output_dir + "/embeddings.npy", embeddings)
    with open(FLAGS.output_dir + "/meta_infos.json", "w") as f:
        json.dump(meta_infos, f)


if __name__ == "__main__":
    app.run(main)
