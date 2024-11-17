import argparse
import fasttext
import tqdm 
import os 

def train_model(input_filename: str,
    model: str = "skipgram",
    dim: int = 100,
    ws: int = 5,
    epoch: int = 5,
    minCount: int = 5,
    loss: str = "ns",
    verbose: int = 2,
    **kwargs
    ) -> fasttext.FastText._FastText:
    return fasttext.train_unsupervised(
        input=input_filename,
        model=model,
        dim=dim,
        ws=ws,
        epoch=epoch,
        minCount=minCount,
        loss=loss,
        verbose=verbose,
        **kwargs
    )


def save_model(model, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f: 
        for word in tqdm.tqdm(model.words):
            vector = model[word]
            vector_str = " ".join(map(str, vector))
            f.write(f"{word} {vector_str}\n")
    print(f"Embeddings saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Train fastText model on provided file.")
    parser.add_argument(
        "input", 
        type=str,
        help="Input .txt file with UTF-8 encoding. One sentence per line."
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Output file to save trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="skipgram",
        help="Model type to be trained (default: 'skipgram')"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        help="Dimensionality of word vectors (default: 100)"
    )
    parser.add_argument(
        "--ws",
        type=int,
        default=5,
        help="Context window size (default: 5)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
        help="Number of epochs for training (default: 5)"
    )
    parser.add_argument(
        "--minCount",
        type=int,
        default=5,
        help="Minimum word frequency to include in the model (default: 5)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ns", "hs", "softmax", "ova"],
        default="ns",
        help="Loss function to use ('ns', 'hs', 'softmax', 'ova'). Default is 'ns'."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Verbosity level during training (0, 1, 2). Default is 2."
    )
    args = parser.parse_args()
    model = train_model(
        input_filename=args.input,
        model=args.model,
        dim=args.dim,
        ws=args.ws,
        epoch=args.epoch,
        minCount=args.minCount,
        loss=args.loss,
        verbose=args.verbose
    )
    save_model(model, args.output)


if __name__ == "__main__":
    main()
