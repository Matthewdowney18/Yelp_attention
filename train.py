import numpy as np
import torch
import torch.utils.data

from dataset import SentenceDataset
from model import Seq2SeqModel, Seq2SeqModelAttention
from utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint


def main():
    nb_epochs = 30
    batch_size = 200
    hidden_size = 256
    embedding_dim = 300
    max_len = 20
    teacher_forcing = 0.6
    min_count = 2
    max_grad_norm = 5
    val_len = 5000
    weight_decay = 0.00001
    model_filename = '/home/mattd/pycharm/yelp/models' \
                     '/baseline_frozen_pretrained'

    eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'
    dataset = SentenceDataset(eng_fr_filename, max_len, min_count)
    print('Dataset: {}'.format(len(dataset)))

    train_len = len(dataset) - val_len
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset, [train_len, val_len])
    print('Train {}, val: {}'.format(len(dataset_train), len(dataset_val)))

    embeddings_dir = '/home/mattd/pycharm/yelp/embeddings.npy'
    embeddings = cuda(get_pretrained_embeddings(embeddings_dir, dataset))

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size, shuffle=False)

    vocab_size = len(dataset.vocab)
    padding_idx = dataset.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset.vocab[SentenceDataset.INIT_TOKEN]
    model = Seq2SeqModel(embeddings, hidden_size, padding_idx, init_idx,
                         max_len, teacher_forcing)
    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab[SentenceDataset.PAD_TOKEN])

    phases = ['train', 'val', ]
    data_loaders = [data_loader_train, data_loader_val, ]

    lowest_loss = 500

    for epoch in range(nb_epochs):
        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = []
            for i, (inputs, targets) in enumerate(data_loader):
                optimizer.zero_grad()

                inputs = variable(inputs)
                targets = variable(targets)

                outputs = model(inputs, targets)

                targets = targets.view(-1)
                outputs = outputs.view(targets.size(0), -1)

                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()

                epoch_loss.append(float(loss))

            epoch_loss = np.mean(epoch_loss)

            if epoch_loss < lowest_loss:
                save_checkpoint(model, loss, optimizer, model_filename)
                lowest_loss = epoch_loss

            if phase == 'train':
                print('Epoch {:03d} | {} loss: {:.3f}'.format(epoch, phase, epoch_loss), end='')
            else:
                print(', {} loss: {:.3f}'.format(phase, epoch_loss), end='\n')

            # print random sentence
            if phase == 'val':
                random_idx = np.random.randint(len(dataset_val))
                inputs, targets = dataset_val[random_idx]
                inputs_var = variable(inputs)

                outputs_var = model(inputs_var.unsqueeze(0)) # unsqueeze to get the batch dimension
                outputs = argmax(outputs_var).squeeze(0).data.cpu().numpy()

                print(u'> {}'.format(get_sentence_from_indices(inputs, dataset.vocab, SentenceDataset.EOS_TOKEN)))
                print(u'= {}'.format(get_sentence_from_indices(targets, dataset.vocab, SentenceDataset.EOS_TOKEN)))
                print(u'< {}'.format(get_sentence_from_indices(outputs, dataset.vocab, SentenceDataset.EOS_TOKEN)))
                print()


if __name__ == '__main__':
    main()