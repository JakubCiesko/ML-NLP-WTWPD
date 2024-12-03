import os
import argparse
import numpy as np

def evaluate(src_emb, tgt_emb, src_lang, tgt_lang, dico_eval):

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Evaluate word translation')
        parser.add_argument("--src_emb", type=str, required=True, help="Path to the source embeddings")
        parser.add_argument("--tgt_emb", type=str, required=True, help="Path to the target embeddings")
        parser.add_argument("--src_lang", type=str, required=True, help="Source language")
        parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
        parser.add_argument("--dico_eval", type=str, required=True, help="Path to the evaluation dictionary")
        args = parser.parse_args()

    evaluate(args.src_emb, args.tgt_emb, args.src_lang, args.tgt_lang, args.dico_eval)