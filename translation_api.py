import torch
import io
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
import math
from torch import Tensor
from torchtext.utils import download_from_url, extract_archive

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Copy the entire PositionalEncoding class from the original file
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

# Copy the entire TransformerModel class from the original file
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
        device = src.device
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

# Tokenizers and Vocabulary (using spaCy)
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Constants (ensure these match your original script)
BATCH_SIZE = 32
D_MODEL = 128
N_HEADS = 8
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LENGTH = 50

# Device configuration
device = torch.device('cpu')

# Function to build vocabulary (similar to original script)
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

# Constants for special tokens
PAD_IDX = fr_vocab['<pad>']
BOS_IDX = fr_vocab['<bos>']
EOS_IDX = fr_vocab['<eos>']

# Create the model
INPUT_DIM = len(fr_vocab)
OUTPUT_DIM = len(en_vocab)

model = TransformerModel(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_HEADS,
                        N_ENCODER_LAYERS, N_DECODER_LAYERS,
                        DIM_FEEDFORWARD, DROPOUT).to(device)

# Load the trained weights
model.load_state_dict(torch.load('transformer_translation.pt', map_location=torch.device('cpu')))
model.eval()

# Translation function
def translate_sentence(sentence: str, max_len=50):
    model.eval()
    
    # Tokenize the source sentence
    tokens = fr_tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    
    src_indexes = [fr_vocab[token] for token in tokens]
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
        tgt_tokens.append(en_vocab.get_itos()[i])
    
    return ' '.join(tgt_tokens)

# FastAPI Application
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request model for input validation
class TranslationRequest(BaseModel):
    french_text: str

# Translation endpoint
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Translate the input French text
        translation = translate_sentence(request.french_text)
        return {"original": request.french_text, "translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Translation service is running"}

# To run the server:
# uvicorn your_file_name:app --reload