from vecto import embeddings
import numpy as np

from dataset import SentenceDataset

def main():
    embeddings_dir = '/mnt/data1/embeddings/crawl/'
    eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'

    dataset = SentenceDataset(eng_fr_filename, 20, 2)

    emb = embeddings.load_from_dir(embeddings_dir)
    vocab_embs = np.zeros((len(dataset.vocab), emb.matrix.shape[1]))

    for i, token in enumerate(dataset.vocab.token2id):
        if emb.has_word(token):
            vocab_embs[i] = emb.get_vector(token)
    np.save('embeddings', vocab_embs)

if __name__ == '__main__':
    main()