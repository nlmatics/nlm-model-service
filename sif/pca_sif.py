import numpy as np
from sklearn.decomposition import PCA

from .embedding_utils import read_h5_embeddings

N = 1

embedding_file = "/home/yi_zhang/data/models/sif/glove.840B.300d.h5"

output_file = f"/home/yi_zhang/data/models/sif/20201013-glove.840B.300d-PCA-{N}.txt"


idx_to_word = []
word2idx = {}


embs = read_h5_embeddings(embedding_file)

dimension = embs.shape[1]
# subtract average vector from each vector
avg_vec = embs.mean(axis=0)
embs = embs - avg_vec


# principal component analysis using sklearn
pca = PCA(svd_solver="full")
pca.fit(embs)


# remove the top N components from each vector
for i in range(len(embs)):
    preprocess_sum = np.zeros(dimension)
    for j in range(N):
        princip = np.array(pca.components_[j])
        preprocess = princip.dot(embs[i])
        preprocess_vec = [princip[k] * preprocess for k in range(len(princip))]
        preprocess_sum = [
            preprocess_sum[k] + preprocess_vec[k] for k in range(len(preprocess_sum))
        ]
    embs[i] = np.array(
        [embs[i][j] - preprocess_sum[j] for j in range(len(preprocess_sum))],
    )


def write_txt():
    with open(output_file, "w", encoding="utf-8") as file:
        # write back new word vector file
        idx = 0
        for idx, vec in enumerate(embs):
            if not isinstance(idx_to_word[idx], str):
                print(idx, idx_to_word[idx])
                continue
            if idx_to_word[idx]:
                file.write(idx_to_word[idx])
                file.write(" ")
                for num in vec:
                    file.write(str(num))
                    file.write(" ")
                file.write("\n")
