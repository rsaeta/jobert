import torch

import os
from pathlib import Path
from transformers import LongformerModel, LongformerTokenizerFast


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def main():
    model_name = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)

    job_files = os.listdir('data')
    results = []
    for fn in job_files:
        with open(Path('data') / fn) as f:
            description = f.read()
            tokens = tokenizer(description, return_tensors='pt')
            outputs = model(**tokens)
            results.append(outputs.pooler_output)

    print(torch.cosine_similarity(results[0], results[1]))
    print('Done')


if __name__ == '__main__':
    main()
