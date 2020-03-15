# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")

train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test_dataset.csv")
sample_submission_path = os.path.join(data_path, 'submit_example.csv')

model_path = os.path.join(data_path, "model")
model_save_path = os.path.join(model_path, "model.bin")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

pretrain_embedding = False
# pretrain_embedding = True
# embed_dim = 300
max_seq_len = {
    '0': 200,
    '1': 250,
    '2': 250
}

# tokenizer = lambda x: x.split(' ')[:max_seq_len]
# padding_idx = 0


num_classes = 3
batch_size = 32
epochs_num = 8

n_splits = 5
train_print_step = 200

label_columns = ['fake_label', 'real_label', 'ncw_label']
