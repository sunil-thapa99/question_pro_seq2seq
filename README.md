# Question Pro - Seq2Seq

## Introduction
The Seq2Seq model in QuestionPro utilizes a robust sequence-to-sequence architecture to generate questions from provided contexts. This model, rooted in deep learning and natural language processing, excels at understanding and transforming input sequences into meaningful output sequences.

With Seq2Seq, the task of question generation becomes dynamic and context-aware. The model can capture intricate relationships within a given context and translate them into well-structured questions. By leveraging the inherent sequential nature of language, Seq2Seq enhances the precision and relevance of the generated questions.

## Dataset and Model
- Link for dataset and model: [Dropbox](https://www.dropbox.com/scl/fo/06z72prw84qvdon24zve7/h?rlkey=92nr17ygw0ghhuies0qwgavvp&dl=0)
- Stanford Question Answering Dataset [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

## Requirements
- Language : Python3.8
- Packages
  -  torch : 1.5.0
  -  spacy : 2.2.4
  -  torchtext : 0.3.1

## Preprocessing Steps

### Case Normalization
Convert all text to lowercase or uppercase to ensure uniformity.

### Tokenization
Break text into individual tokens (words or subword units) to facilitate processing.

### Named Entity Recognition (NER)
Identify and classify entities (e.g., names, locations) within the text.

### POS-Tagging (Part-of-Speech Tagging)
Assign grammatical parts of speech to each token in the text.

### IOB-Tagging (Inside-Outside-Beginning Tagging)
Label tokens with Inside, Outside, or Beginning tags to represent entities sequentially.

## Pairing Input and Output
Organize the preprocessed data into input-output pairs for training the Seq2Seq model.

These preprocessing steps lay the foundation for training a powerful Seq2Seq model by ensuring the input data is appropriately formatted and enriched with linguistic information. Each step is crucial in preparing the data for effective sequence-to-sequence processing.


## API test

```
/seq2seq
{
  "context": " ",
  "answer": " ",
  "answer_start": [int]
  }
```
