import spacy
import pandas as pd

# build_vocab  data
import torchtext
from torchtext.data import Field, TabularDataset
import spacy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
import random
import pandas as pd
import numpy as np

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score

import os
DIR = os.getcwd()

# Load spacy models
spacy_en = spacy.load('en_core_web_md')

def extract_features(text, answer, answer_start, nlp):
    '''
    Extract answers to obtain POS, NER, case, BIO features based on text

    Arguments:
        text	-- context or paragraph
        answer 	-- answer in paragraph's question
        answer_start -- starting index of answer
        nlp 	-- spacy tool for nlp
    Returns:
        pos 	-- sequence of string of answer tokens part-of-speech tagging
		ner 	-- sequence of string of answer tokens named entity recognition
		case	-- sequence of string of answer tokens case
		bio 	-- sequence of string of answer tokens inside-outside-beggining tagging
		tokenized 	-- joined tokenized context (paragraph) with lower typecasting
    '''
    
    # Extract answer location index (left, right and answers itself) in text
    left = text[0:answer_start]
    ans = text[answer_start:answer_start+len(answer)+1]
    right = text[answer_start+len(answer)+1:len(text)+1]    
    
    # Initialize return values list
    pos = []
    ner = []
    case = []
    bio = []
    tokenized = []
    
    left_side = nlp(left)
    answer_range = nlp(ans)
    right_side = nlp(right)
    
    for token in left_side:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            bio.append('O')
    
    for token in answer_range:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            if token.i == 0:
                bio.append('B')
            else:
                bio.append('I')
    
    for token in right_side:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            bio.append('O')
                
    return (' '.join(pos)), (' '.join(ner)), (' '.join(case)), (' '.join(bio)), (' '.join(tokenized))

def build_lexical_features(data):
    '''
    Creating pandas dataframe of features from parsed data

    Arguments:
        data -- data to be extracted; data must have context, answer, answer_start and question column
    Returns:
        data -- pandas dataframe of questions, context and features: IOB tag and lexical features(POS tag, NER, and case). 
    '''
    data['BIO'] = ''
    data['LEX'] = ''
    count = 0
    for idx, text, answer, answer_start, question in data[['context', 'answer', 'answer_start','question']].itertuples():
        print(text)
        pos, ner, case, data['BIO'][idx], data['context'][idx] = extract_features(text, str(answer), int(answer_start), spacy_en)
        lex = [i + '_' + j + '_' + k for i, j, k in zip(pos.split(), ner.split(), case.split())]
        data['LEX'][idx] = ' '.join(lex)
        data['question'][idx] = ' '.join([token.text.lower() for token in spacy_en(question)])
        count+=1
        print(count)

    # Building data on selected columns
    data = data[['context', 'question', 'BIO', 'LEX']]

    return data

class Encoder(nn.Module):

    def __init__(self, hidden_size, embedding_size,
                 embedding, answer_embedding, lexical_embedding, n_layers, dropout):

        super(Encoder, self).__init__()

        # Initialize network parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Embedding layer to be shared with Decoder
        self.embedding = embedding
        self.answer_embedding = answer_embedding
        self.lexical_embedding = lexical_embedding

        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=n_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, input_sequence, input_lengths, answer_sequence, lexical_sequence):

        # Convert input_sequence to word embeddings
        word_embeddings = self.embedding(input_sequence)
        answer_embeddings = self.answer_embedding(answer_sequence)
        lexical_embeddings = self.lexical_embedding(lexical_sequence)

        # Concatenate word embeddings from all features
        final_embeddings = torch.cat((word_embeddings,answer_embeddings,lexical_embeddings), 0)

        # Pack the sequence of embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(final_embeddings, input_lengths)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # The ouput of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

    def dot_score(self, hidden_state, encoder_states):
        # Attention model use the dot product formula as global attention
        return torch.sum(hidden_state * encoder_states, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()

        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, embedding, embedding_size,
                 hidden_size, output_size, n_layers, dropout):

        super(Decoder, self).__init__()

        # Initialize network params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)

    def forward(self, current_token, hidden_state, encoder_outputs, mask):

        # convert current_token to word_embedding
        embedded = self.embedding(current_token)

        # Pass through GRU
        rnn_output, hidden_state = self.gru(embedded, hidden_state)

        # Calculate attention weights
        attention_weights = self.attn(rnn_output, encoder_outputs, mask)

        # Calculate context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate  context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Pass concat_output to final output layer
        output = self.out(concat_output)

        # Return output and final hidden state
        return output, hidden_state

class Seq2seq(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size,
                 device, pad_idx, eos_idx, sos_idx, teacher_forcing_ratio=0.5):
        super(Seq2seq, self).__init__()

        # Initialize embedding layer shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.answer_embedding = nn.Embedding(6, embedding_size, padding_idx=1)
        # Size could sometime change, depend on the device that the model is trained on
        self.lexical_embedding = nn.Embedding(452, embedding_size, padding_idx=1)

        # Encoder network
        self.encoder = Encoder(hidden_size,
                               embedding_size,
                               self.embedding,
                               self.answer_embedding,
                               self.lexical_embedding,
                               n_layers=2,
                               dropout=0.5)

        # Decoder network
        self.decoder = Decoder(self.embedding,
                               embedding_size,
                               hidden_size,
                               vocab_size,
                               n_layers=2,
                               dropout=0.5)


        # Indices of special tokens and hardware device
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.device = device

    def create_mask(self, input_sequence):

        return (input_sequence != self.pad_idx).permute(1, 0)

    def forward(self, input_sequence, answer_sequence, lexical_sequence, output_sequence, teacher_forcing_ratio):

        # Unpack input_sequence tuple
        input_tokens = input_sequence[0]
        input_lengths = input_sequence[1]

        # Unpack output_tokens, or create an empty tensor for text generation
        if output_sequence is None:
            inference = True
            output_tokens = torch.zeros((100, input_tokens.shape[1])).long().fill_(self.sos_idx).to(self.device)
        else:
            inference = False
            output_tokens = output_sequence[0]

        vocab_size = self.decoder.output_size

        batch_size = len(input_lengths)
        max_seq_len = len(output_tokens)

        # Tensor initialization to store Decoder output
        outputs = torch.zeros(max_seq_len, batch_size, vocab_size).to(self.device)

        # Pass through the first half of the network
        encoder_outputs, hidden = self.encoder(input_tokens, input_lengths, answer_sequence, lexical_sequence)

        # Ensure dim of hidden_state can be fed into Decoder
        hidden =  hidden[:self.decoder.n_layers]

        # First input to the decoder is the <sos> tokens
        output = output_tokens[0,:]

        # Create mask
        mask = self.create_mask(input_tokens)

        # Step through the length of the output sequence one token at a time
        # Teacher forcing is used to assist training
        for t in range(1, max_seq_len):
            output = output.unsqueeze(0)

            output, hidden = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (output_tokens[t] if teacher_force else top1)

            # If we're in inference mode, keep generating until we produce an
            # <eos> token
            if inference and output.item() == self.eos_idx:
                return outputs[:t]

        return outputs


# Set random seeds for reproducibility
SEED = 1234

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
random.seed(SEED)
np.random.seed(SEED)

# Load data
resume = os.path.join('models/model_14.pth')

# Create Field object
tokenize = lambda x: x.split()
TEXT = Field(tokenize=tokenize, lower=False, include_lengths = True, init_token = '<SOS>', eos_token = '<EOS>')
LEX = Field(tokenize=tokenize, lower=False, init_token = '<SOS>', eos_token = '<SOS>')
BIO = Field(tokenize=tokenize, lower=False, init_token = '<SOS>', eos_token = '<SOS>')

# Specify Fields in the dataset
fields = [('context', TEXT), ('question', TEXT), ('bio', BIO), ('lex', LEX)]

# Build vocabulary
MAX_VOCAB_SIZE = 35000
MIN_COUNT = 5

vocab_dir = os.path.join(DIR, 'vocabs')

# Load the vocabularies
text_vocab = torch.load(os.path.join(vocab_dir, 'text_vocab.pth'))
bio_vocab = torch.load(os.path.join(vocab_dir, 'bio_vocab.pth'))
lex_vocab = torch.load(os.path.join(vocab_dir, 'lex_vocab.pth'))

# Assign the loaded vocabularies to your fields
TEXT.vocab = text_vocab
BIO.vocab = bio_vocab
LEX.vocab = lex_vocab

pad_idx = 1
eos_idx = 3
sos_idx = 2

# Size of embedding_dim should match the dim of pre-trained word embeddings
embedding_dim = 300
hidden_dim = 512
vocab_size = len(TEXT.vocab)

# Initializing weights
model = Seq2seq(embedding_dim, hidden_dim, vocab_size, device, pad_idx, eos_idx, sos_idx).to(device)

# Load model
model.load_state_dict(torch.load(resume, map_location=torch.device('cpu')))


def predict_question(model, paragraph, answer_pos, lex_features):
    model.eval()

    tokenized = ['<SOS>'] + paragraph + ['<EOS>']
    numericalized = [TEXT.vocab.stoi[t] for t in tokenized]

    tokenized_answer = ['<SOS>'] + answer_pos + ['<EOS>']
    numericalized_answer = [BIO.vocab.stoi[t] for t in tokenized_answer]

    tokenized_lex = ['<SOS>'] + lex_features + ['<EOS>']
    numericalized_lex = [LEX.vocab.stoi[t] for t in tokenized_lex]

    paragraph_length = torch.LongTensor([len(numericalized)]).to(model.device)
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(model.device)

    answer_tensor = torch.LongTensor(numericalized_answer).unsqueeze(1).to(model.device)
    lex_tensor = torch.LongTensor(numericalized_lex).unsqueeze(1).to(model.device)

    question_tensor_logits = model((tensor, paragraph_length), answer_tensor, lex_tensor, None, 0)

    question_tensor = torch.argmax(question_tensor_logits.squeeze(1), 1)
    question = [TEXT.vocab.itos[t] for t in question_tensor]

    # Start at the first index.  We don't need to return the <SOS> token
    question = question[1:]

    return question, question_tensor_logits