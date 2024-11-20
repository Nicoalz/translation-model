import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math
import time
from typing import Tuple
import io
from torchtext.utils import download_from_url, extract_archive
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

import torch
print(torch.__version__)


# Device configuration
device = torch.device('cpu' if torch.backends.mps.is_available() else 'cpu')
# print(f"Using device: {device}")

# Download and extract the dataset
url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.fr.gz', 'train.en.gz')
val_urls = ('val.fr.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# Tokenizers
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Build vocabulary
def build_vocab(filepath, tokenizer):
    def yield_tokens(file_path):
        with io.open(file_path, encoding="utf8") as f:
            for line in f:
                yield tokenizer(line.rstrip('\n'))
                
    vocab = build_vocab_from_iterator(yield_tokens(filepath),
                                    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
                                    special_first=True)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

fr_vocab = build_vocab(train_filepaths[0], fr_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

# Constants
BATCH_SIZE = 32
PAD_IDX = fr_vocab['<pad>']
BOS_IDX = fr_vocab['<bos>']
EOS_IDX = fr_vocab['<eos>']
MAX_LENGTH = 50

# Data processing
def data_process(filepaths):
    raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
        fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr.rstrip('\n'))],
                                dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en.rstrip('\n'))],
                                dtype=torch.long)
        if len(fr_tensor_) <= MAX_LENGTH and len(en_tensor_) <= MAX_LENGTH:
            data.append((fr_tensor_, en_tensor_))
    return data

# Prepare datasets
train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

def generate_batch(data_batch):
    fr_batch, en_batch = [], []
    for (fr_item, en_item) in data_batch:
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return fr_batch, en_batch

# Create data iterators
train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iterator = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=False)
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.src_mask = None
        self.tgt_mask = None
        self.src_padding_mask = None
        self.tgt_padding_mask = None
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask,
                                None, src_padding_mask, tgt_padding_mask, None)
        return self.fc_out(output)

# Model parameters
INPUT_DIM = len(fr_vocab)
OUTPUT_DIM = len(en_vocab)
D_MODEL = 128
N_HEADS = 8
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Initialize model
model = TransformerModel(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_HEADS,
                        N_ENCODER_LAYERS, N_DECODER_LAYERS,
                        DIM_FEEDFORWARD, DROPOUT).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training function
def train_epoch(model, iterator, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    iterations_per_epoch = len(iterator)
    for i, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt[:-1])
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        tgt = tgt[1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Batch: {i+1} / {iterations_per_epoch} | Loss: {loss.item():.4f}')
    
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:-1])
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Training loop
def train_model(model, train_iterator, valid_iterator, optimizer, criterion, N_EPOCHS):
    best_valid_loss = float('inf')
    writer = SummaryWriter('runs/transformer_translation')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        # Log losses to TensorBoard
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', valid_loss, epoch)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_path = f'transformer_translation_{epoch}.pt'
            torch.save(model.state_dict(), model_path)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Translation function
def translate_sentence(model, sentence: str, src_tokenizer, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    
    # Tokenize the source sentence
    tokens = src_tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    
    src_indexes = [src_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    tgt_indexes = [BOS_IDX]
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(1).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        pred_token = output.argmax(2)[-1, 0].item()
        tgt_indexes.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    tgt_tokens = []
    for i in tgt_indexes[1:]:  # Skip the <bos> token
        if i == EOS_IDX:
            break
        tgt_tokens.append(tgt_vocab.get_itos()[i])
    
    return ' '.join(tgt_tokens)

# Training the model (uncomment to train)

# N_EPOCHS = 5
# print("Starting Training...")
# train_model(model, train_iterator, valid_iterator, optimizer, criterion, N_EPOCHS)
# print("Training completed!")


# Loading the model (uncomment to load a trained model)

model.load_state_dict(torch.load('transformer_translation.pt'))
print("Model loaded successfully!")


# Test translation
def test_translation(sentence):
    model.eval()
    translation = translate_sentence(model, sentence, fr_tokenizer, fr_vocab, en_vocab, device)
    print(f'\nFrench: {sentence}')
    print(f'English: {translation}')

# Example usage (uncomment to test)

# test_sentence = "Cinq femmes marchent dans la rue"
# test_translation(test_sentence)



import torch 

# Define dummy inputs matching your model's input dimensions
dummy_src = torch.randint(0, len(fr_vocab), (MAX_LENGTH, BATCH_SIZE)).to(device)  # src tensor
dummy_tgt = torch.randint(0, len(en_vocab), (MAX_LENGTH, BATCH_SIZE)).to(device)  # tgt tensor

# Specify the file path for the ONNX model
onnx_file_path = "transformer_translation.onnx"

# Export the model to ONNX
torch.onnx.export(
    model, 
    (dummy_src, dummy_tgt),  # Input tensors as a tuple
    onnx_file_path, 
    export_params=True,  # Store the trained parameter weights
    opset_version=14,  # Use opset version 14
    do_constant_folding=True,  # Optimize constant folding for inference
    input_names=["src", "tgt"],  # Name input layers
    output_names=["output"],  # Name output layers
    dynamic_axes={
        "src": {0: "sequence_length", 1: "batch_size"},  # Variable sequence length & batch size
        "tgt": {0: "sequence_length", 1: "batch_size"},
        "output": {0: "sequence_length", 1: "batch_size"},
    },
    verbose=True  # Enable detailed logging
)


print(f"Model successfully exported to {onnx_file_path}")
