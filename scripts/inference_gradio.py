import gradio as gr
import torch
from unixcoder import UniXcoder
import numpy as np
import json
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("owner", None, "GitHub Owner", required=True)
flags.DEFINE_string("input_dir", None, "Input directory", required=True)
flags.DEFINE_string("model_name", "microsoft/unixcoder-base", "Model name")
flags.DEFINE_integer("max_length", 512, "Max length of the input sequence")


def main(argv):
    embeddings = np.load(FLAGS.input_dir + "/embeddings.npy")
    with open(FLAGS.input_dir + "/meta_infos.json", "r") as f:
        meta_infos = json.load(f)
    embeddings = torch.tensor(embeddings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(device)

    def find_most_similar(query):
        tokens_ids = model.tokenize([query], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)

        with torch.no_grad():
            _, nl_embedding = model(source_ids)
            nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

            scores = torch.mm(embeddings, nl_embedding.t())
            topk = torch.squeeze(
                torch.argsort(scores, dim=0, descending=True)[:10], dim=1
            )
            scores = torch.squeeze(scores, dim=1)

        scores = scores.cpu().detach().numpy()
        topk = topk.cpu().detach().numpy()

        result = "# Search Result\n\n\n"
        for cnt, i in enumerate(topk):
            repo_name = meta_infos[i]["repo_name"]
            ref = meta_infos[i]["ref"]
            file = meta_infos[i]["file"]
            line = meta_infos[i]["line"]
            end_line = meta_infos[i]["end_line"]
            result += f"## #{cnt}\n"
            result += f"\n"
            result += f"* Link: [{FLAGS.owner}/{repo_name}/{file}](https://github.com/{FLAGS.owner}/{repo_name}/blob/{ref}/{file}#L{line}-L{end_line})\n"
            result += f"* Score: {scores[i]}\n"
            result += f"\n"

        return result

    gr.Interface(
        fn=find_most_similar,
        inputs=gr.Textbox(label="Query"),
        outputs=["markdown"],
        examples=[
            "",
        ],
        allow_flagging="never",
    ).launch()


if __name__ == "__main__":
    app.run(main)
