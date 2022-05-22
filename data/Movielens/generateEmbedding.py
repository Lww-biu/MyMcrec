import torch
import torch.nn as nn
import torch.nn.functional as F


def generateEmbedding(ffile, ssize):
    llatent = UIEmbedding(64, ssize)
    input = torch.LongTensor(range(1, ssize + 1))
    iinput = llatent(input)
    for i in range(0, ssize):
        line = str(i) + " "
        line = line + " ".join(map(str, iinput[i].detach().tolist())) + "\n"
        ffile.write(line)


class UIEmbedding(nn.Module):

    def __init__(self, latent_dim, obj_num):
        super(UIEmbedding, self).__init__()

        self.latent_dim = latent_dim
        # id starts from 1, add one more id 0 for invalid updates
        self.embedding = nn.Embedding(num_embeddings=obj_num + 1,
                                      embedding_dim=latent_dim)

        nn.init.xavier_normal_(self.embedding.weight.data)

    def forward(self, input):
        # input.shape: batch_size, negative_num + 1, latent_dim

        input = self.embedding(input.long())
        input = input.view(-1, self.latent_dim)

        return input


if __name__ == "__main__":
    latent_Dim = 64
    u_size = 943
    i_size = 1682
    g_size = 18
    emb_size = [u_size, i_size, g_size]
    emb_file = [
        "user_node_emb2.dat", "item_node_emb2.dat", "genre_node_emb2.dat"
    ]
    for i in range(3):
        ffile = open(emb_file[i], "w", encoding="utf-8")
        generateEmbedding(ffile, emb_size[i])
        ffile.close()
