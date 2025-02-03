import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # parser.add_argument("--n_head", type=int, default=8, help="GPT number of heads")
    parser.add_argument("--n_layer", type=int, default=12, help="GPT number of layers")

    # parser.add_argument("--n_batch", type=int, default=512, help="Batch size")
  
    # parser.add_argument(
    #     "--lr_start", type=float, default=3 * 1e-4, help="Initial lr value"
    # )
    # parser.add_argument(
    #     "--lr_end", type=float, default=3 * 1e-4, help="Maximum lr weight value"
    # )

    # parser.add_argument(
    #     "--max_len", type=int, default=100, help="Max of length of SMILES"
    # )
    parser.add_argument(
        "--max_epochs", type=int, required=False, default=1, help="max number of epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--smiles_graph_weight", type=float, default=0.3333)
    parser.add_argument("--smiles_fp_weight", type=float, default=0.3333)
    parser.add_argument("--graph_fp_weight", type=float, default=0.3333)
    parser.add_argument(
        "--save_model_folder", type=str, required=False, default="save_model"
    )
    parser.add_argument(
        "--graph_model", type=str, required=False, default="bi_bi_graph"
    )
    parser.add_argument(
        "--model_path", type=str, required=False, default="bi_bi_graph"
    )

    parser.add_argument(
        "--test", type=bool, required=False, default=False
    )

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
