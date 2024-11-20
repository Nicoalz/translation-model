# You can now load the package via spacy.load('fr_core_news_sm')

# 1
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
import math
import time
import spacy

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.fr.gz', 'train.en.gz')
val_urls = ('val.fr.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    
    # Create vocabulary from counter and define special tokens directly using vocab
    vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


fr_vocab = build_vocab(train_filepaths[0], fr_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
  raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
    fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
    data.append((fr_tensor_, en_tensor_))
  return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

# 2


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
print('Device:', device)

BATCH_SIZE = 128
PAD_IDX = fr_vocab['<pad>']
BOS_IDX = fr_vocab['<bos>']
EOS_IDX = fr_vocab['<eos>']


def generate_batch(data_batch):
  fr_batch, en_batch = [], []
  for (fr_item, en_item) in data_batch:
    fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return fr_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

# 3
# error of nan:
# class Encoder(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  emb_dim: int,
#                  enc_hid_dim: int,
#                  dec_hid_dim: int,
#                  dropout: float):
#         super().__init__()

#         self.input_dim = input_dim
#         self.emb_dim = emb_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.dropout = dropout

#         self.embedding = nn.Embedding(input_dim, emb_dim)

#         self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

#         self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self,
#                 src: Tensor) -> Tuple[Tensor]:

#         embedded = self.dropout(self.embedding(src))

#         outputs, hidden = self.rnn(embedded)

#         hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

#         return outputs, hidden


# class Attention(nn.Module):
#     def __init__(self,
#                  enc_hid_dim: int,
#                  dec_hid_dim: int,
#                  attn_dim: int):
#         super().__init__()

#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim

#         self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

#         self.attn = nn.Linear(self.attn_in, attn_dim)

#     def forward(self,
#                 decoder_hidden: Tensor,
#                 encoder_outputs: Tensor) -> Tensor:

#         src_len = encoder_outputs.shape[0]

#         repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

#         encoder_outputs = encoder_outputs.permute(1, 0, 2)

#         energy = torch.tanh(self.attn(torch.cat((
#             repeated_decoder_hidden,
#             encoder_outputs),
#             dim = 2)))

#         attention = torch.sum(energy, dim=2)

#         return F.softmax(attention, dim=1)


# class Decoder(nn.Module):
#     def __init__(self,
#                  output_dim: int,
#                  emb_dim: int,
#                  enc_hid_dim: int,
#                  dec_hid_dim: int,
#                  dropout: int,
#                  attention: nn.Module):
#         super().__init__()

#         self.emb_dim = emb_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.output_dim = output_dim
#         self.dropout = dropout
#         self.attention = attention

#         self.embedding = nn.Embedding(output_dim, emb_dim)

#         self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

#         self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

#         self.dropout = nn.Dropout(dropout)


#     def _weighted_encoder_rep(self,
#                               decoder_hidden: Tensor,
#                               encoder_outputs: Tensor) -> Tensor:

#         a = self.attention(decoder_hidden, encoder_outputs)

#         a = a.unsqueeze(1)

#         encoder_outputs = encoder_outputs.permute(1, 0, 2)

#         weighted_encoder_rep = torch.bmm(a, encoder_outputs)

#         weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

#         return weighted_encoder_rep


#     def forward(self,
#                 input: Tensor,
#                 decoder_hidden: Tensor,
#                 encoder_outputs: Tensor) -> Tuple[Tensor]:

#         input = input.unsqueeze(0)

#         embedded = self.dropout(self.embedding(input))

#         weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
#                                                           encoder_outputs)

#         rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

#         output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

#         embedded = embedded.squeeze(0)
#         output = output.squeeze(0)
#         weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

#         output = self.out(torch.cat((output,
#                                      weighted_encoder_rep,
#                                      embedded), dim = 1))

#         return output, decoder_hidden.squeeze(0)


# class Seq2Seq(nn.Module):
#     def __init__(self,
#                  encoder: nn.Module,
#                  decoder: nn.Module,
#                  device: torch.device):
#         super().__init__()

#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device

#     def forward(self,
#                 src: Tensor,
#                 trg: Tensor,
#                 teacher_forcing_ratio: float = 0.5) -> Tensor:

#         batch_size = src.shape[1]
#         max_len = trg.shape[0]
#         trg_vocab_size = self.decoder.output_dim

#         outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

#         encoder_outputs, hidden = self.encoder(src)
#         # first input to the decoder is the <sos> token
#         if torch.isnan(encoder_outputs).any() or torch.isinf(encoder_outputs).any():
#             print(encoder_outputs)
#             print(f"NaN or Inf detected in encoder_outputs")

#         output = trg[0,:]

#         for t in range(1, max_len):
#             output, hidden = self.decoder(output, hidden, encoder_outputs)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.max(1)[1]
#             output = (trg[t] if teacher_force else top1)

#         return outputs

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        # Projection layer to match decoder's hidden state dimension
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        # Combine forward and backward hidden states and project to decoder's hidden size
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 n_heads: int = 4):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Use MultiheadAttention directly in the class
        self.attention = nn.MultiheadAttention(
            embed_dim=dec_hid_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=False
        )

        # Project encoder outputs to match decoder's hidden dimension
        self.encoder_proj = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.rnn = nn.GRU(emb_dim + dec_hid_dim, dec_hid_dim)

        self.out = nn.Linear(dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        # Project encoder outputs to match decoder's hidden dimension
        encoder_outputs_projected = self.encoder_proj(encoder_outputs)

        # Compute attention
        attn_output, _ = self.attention(
            query=decoder_hidden.unsqueeze(0),  
            key=encoder_outputs_projected,     
            value=encoder_outputs_projected
        )

        rnn_input = torch.cat((embedded, attn_output), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        output = output.squeeze(0)
        output = self.out(output)

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(fr_vocab)
OUTPUT_DIM = len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5 
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32  
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Create the encoder
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

# Create the attention mechanism first
attn = nn.MultiheadAttention(
    embed_dim=DEC_HID_DIM, 
    num_heads=4, 
    dropout=DEC_DROPOUT,
    batch_first=False
)

# Create the decoder, passing the attention mechanism
dec = Decoder(
    output_dim=OUTPUT_DIM, 
    emb_dim=DEC_EMB_DIM, 
    enc_hid_dim=ENC_HID_DIM, 
    dec_hid_dim=DEC_HID_DIM, 
    dropout=DEC_DROPOUT
)


# Create the seq2seq model
model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.constant_(param.data, 0)



model.apply(init_weights)

model = model.float()


optimizer = optim.AdamW(model.parameters(), lr=0.0005)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_IDX = en_vocab.get_stoi()['<pad>']


criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0
    total_iterations = len(iterator)

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(output)
            print(f"NaN or Inf detected in output at batch {i + 1}")
            break
        if torch.isnan(trg).any() or torch.isinf(trg).any():
            print(f"NaN or Inf detected in target at batch {i + 1}")
            break


        loss = criterion(output, trg)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN or Inf detected in loss at batch {i + 1}")
            break

        loss.backward()
        
        # Check for NaN or Inf gradients
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"NaN or Inf detected in gradients for {name}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        
        # Print iteration progress (current iteration/total iterations)
        if (i + 1) % 10 == 0:  # Print every 10 iterations, you can adjust this as needed
            print(f"Iteration: {i + 1}/{total_iterations} | Loss: {loss.item():.4f}")

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1
CLIP = 1


best_valid_loss = float('inf')
print('Training started')
for epoch in range(N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time()
    print('Start time:', start_time)
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    print('Train loss:', train_loss)
    valid_loss = evaluate(model, valid_iter, criterion)
    print('Valid loss:', valid_loss)

    end_time = time.time()
    print('End time:', end_time)

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Save the model
torch.save(model.state_dict(), 'seq2seq_model.pt')
print('Model saved')

# # Load the model
# model.load_state_dict(torch.load('seq2seq_model.pt'))
# model.eval()
# print('Model loaded')

# translation function

def translate_sentence(sentence, model, fr_vocab, en_vocab, fr_tokenizer, max_output_length=50):
    model.eval()
    
    # Tokenize and convert to tensor
    tokens = fr_tokenizer(sentence)
    tokens = [fr_vocab['<bos>']] + [fr_vocab[token] for token in tokens] + [fr_vocab['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    
    # Encode the source sentence
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    # Initialize decoder input and outputs
    trg_indexes = [en_vocab['<bos>']]
    
    for _ in range(max_output_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        # Get the most likely next token
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        # Stop if end of sentence token is predicted
        if pred_token == en_vocab['<eos>']:
            break
    
    # Convert tokens back to words
    trg_tokens = [en_vocab.get_itos()[i] for i in trg_indexes[1:-1]]
    
    return ' '.join(trg_tokens)

# Example usage
# def test_translation(french_sentence):
#     translated = translate_sentence(
#         sentence=french_sentence, 
#         model=model, 
#         fr_vocab=fr_vocab, 
#         en_vocab=en_vocab, 
#         fr_tokenizer=fr_tokenizer
#     )
#     print(f"French: {french_sentence}")
#     print(f"English: {translated}")

# test_translation("Cinq femmes marchent dans la rue")