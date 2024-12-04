import os
import argparse
import numpy as np
import torch
import torch.nn as nn

def evaluate(src_emb, tgt_emb, src_lang, tgt_lang, dico_eval):

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Evaluate word translation')
        parser.add_argument("--src_emb", type=str, required=True, help="Path to the source embeddings")
        parser.add_argument("--tgt_emb", type=str, required=True, help="Path to the target embeddings")
        parser.add_argument("--src_lang", type=str, required=True, help="Source language")
        parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
        parser.add_argument("--dico_eval", type=str, required=True, help="Path to the evaluation dictionary")
        args = parser.parse_args()

    # check parameters

    # build logger / model / trainer / evaluator

    # run evaluations
    evaluate(args.src_emb, args.tgt_emb, args.src_lang, args.tgt_lang, args.dico_eval)

class Discriminator(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=2048, dropout_rate=0.1, smoothing_coeff=0.2, leaky_relu_slope=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.smoothing_coeff = smoothing_coeff

    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs