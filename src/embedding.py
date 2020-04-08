import torch as tr
import torch.nn as nn

in_channels = 5

class NucleotideEmbedding(nn.Module):
    def __init__(self):
        super(NucleotideEmbedding, self).__init__()
        weight = tr.Tensor([[ 1, 0, 0, 0, 0],
                            [ 0, 1, 0, 0, 0],
                            [ 0, 0, 1, 0, 0],
                            [ 0, 0, 0, 1, 0],
                            [ 1, 0, 0, 1, 0],
                            [ 0, 1, 1, 0, 0],
                            [ 0, 0, 1, 1, 0],
                            [ 1, 1, 0, 0, 0],
                            [ 0, 1, 0, 1, 0],
                            [ 1, 0, 1, 0, 0],
                            [ 0, 1, 1, 1, 0],
                            [ 1, 0, 1, 1, 0],
                            [ 1, 1, 0, 1, 0],
                            [ 1, 1, 1, 0, 0],
                            [ 1, 1, 1, 1, 0]])
        weight /= weight.sum(1, keepdim=True)
        weight = tr.cat([weight, weight, weight])
        weight[15:30, 4] =  1
        weight[30:45, 4] = -1
        weight = tr.cat([tr.Tensor([[0,0,0,0,0]]), weight])
        self.embedding = nn.Embedding.from_pretrained(weight)

    def forward(self, x):
        x = x.to(tr.int64)
        x = self.embedding(x)
        x = x.transpose(1,2)
        return x

