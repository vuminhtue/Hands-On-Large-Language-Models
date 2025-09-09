# Chapter 2: Tokens and Embeddings - Homework Assignment

**Course:** Large Language Models

**Chapter:** Tokens and Embeddings

**Total Points:** 100 points

**Due Date:** One week from the published date

---

## Instructions

This homework assignment tests your understanding of tokenization and embeddings, which are foundational concepts for Large Language Models. Answer all questions completely and show your work where applicable.

---

## Part I: Descriptive Questions (20 points)

### Question 1 (5 points)
Explain the role of a tokenizer in the LLM pipeline. Why is it necessary to convert text into token IDs before the model can process it? Describe the difference between the input the user provides and the input the model actually receives.

### Answer 1:
Tokenizers are the first step in processing input language to an LLM. In an LLM pipeline, the tokenizer transform raw text into a sequence of discrete token (subword/byte pieces) and maps each token to an integer ID in the model’s vocabulary. On average, a token is often 3/4 of a word.

Tokenization is a necessary step to turn raw text into numbers a neural network can process. Each LLM ships with its own tokenizer and vocabulary size; they aren’t interchangeable.

A user’s input may include text (and sometimes files, images, audio), punctuation, and emoji. The model, however, receives token IDs (plus special tokens and metadata like attention masks/position info)—a numeric, neural-network-friendly representation.
In short: users write strings; models see token IDs.

### Question 2 (5 points)
Compare and contrast the four main tokenization methods discussed in the chapter: word, subword, character, and byte tokens. What are the primary advantages and disadvantages of each, especially concerning vocabulary size and handling of unknown words?

### Answer 2:

There are multiple tokenization scheme exist, including word, subword tokens, characters or bytes depending on the specific requirement of a given application. The comparison is as follow table:

![alt text](https://github.com/vuminhtue/Hands-On-Large-Language-Models/blob/main/chapter02/tokenize.JPG?raw=true)

| Method                                        | CRISP (ratings)                                           | Unknown / rare words                                            |Vocab size            | Advantage                                                                        | Disadvantage                                                          |
| --------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Word tokens** (whole words)                 | **C=2, R=2, I=2, S=3, P=4**                                       | Poor: unseen items tokenized to **[UNK]**.             | Large (OOV risk)              | Short sequences and efficient when words are in-vocab.                           | Sensitive to typos and other languages; semantic lost on OOV.      |
| **Subword tokens** (byte-level BPE / Unigram) | **C=5, R=4, I=4, S=3, P=4**                                       | Strong: no **[UNK]**; rare words span across a few tokens.     | Moderate  | Best overall trade-off: broad coverage with efficient sequences   | Can over-split domain; sequence length inflates.  |
| **Character tokens** (unicode chars)          | **C=5, R=3, I=2, S=2, P=2**   | Excellent: every character representable (no **[UNK]**).          | Small–to–medium        | Maximum coverage and reversibility.                                              | Very long sequences; weaker local semantics → higher compute.         |
| **Byte tokens** (raw bytes)                   | **C=5, R=3, I=2, S=2, P=2**    | Perfect coverage; nothing OOV—inputs may split into **bytes**.  | Very small                   | Universal and reversible across languages/symbols.                               | Highly fragmented units; inefficient.       |



### Question 3 (5 points)
What is the difference between a static token embedding (like those from word2vec) and a contextualized word embedding produced by a modern LLM like DeBERTa? Use an example to illustrate why context is important for word representation.

### Answer 3:
- Static token embedding (word2vec) apply fix token ID in its vocabulary to the same word despite their different semantic meaning in context
- Contextualized embedding produced by modern LLM such as BERT based family has flexible/variable token ID to the same word that has different semantic meaning in different context. The self-attention mechanism help to change the token ID (represented by vector) based on the surrounding context.
- Example: the word **bank** in **river bank** is different from the **money in the bank**

### Question 4 (5 points)
Describe the core principle behind the word2vec algorithm. Explain the roles of "positive examples" (neighboring words) and "negative examples" (non-neighboring words) in the contrastive training process.

### Answer 4:

![alt text](https://github.com/vuminhtue/Hands-On-Large-Language-Models/blob/main/chapter02/word2vec.JPG?raw=true)

- In word2vec algorithm's core principle, words that occur in similar semantic context should have similar vectors (neighbouring), and on the other hand, words not having same context will have very low relation.
- In this method, fixed continuos vector is affix to each word/subword type. The training goal is to find vector that have high probability of being real neighbour (positive examples) with target label 1. In addition, some dummy words (not neighbouring), which are sampled from a noise distribution, are called negative example also used in the training process (with target label 0).
- For each sentence, we apply a sliding window to setup positive pair (w,c) from target word - vector **u<sub>w</sub>** and context word - vector **v<sub>c</sub>**. For each positive pair, sample K (5-20) negative context **v<sub>ni</sub>** from a noise unigram distribution, they represent for non-neighbouring words for target word - vector **u<sub>w</sub>**. The objective function is to have high the dot product for positive pairs and low dot product for negative pairs. After the training, each word has a static embedding vector which captures the semantic similarity via context.
---

## Part II: Multiple Choice Questions (20 points)

**Instructions:** Choose the best answer for each question. Each question is worth 2 points.

### Question 5
What is the typical output of an LLM tokenizer that is fed to the language model?

A) A string of cleaned text.

B) A list of floating-point numbers representing embeddings.

C) A list of integers representing token IDs.

D) A dictionary of word counts.

### Answer 5: **C**

### Question 6
Which tokenization method is most commonly used in modern LLMs like GPT-4 and StarCoder2?

A) Word tokenization

B) Character tokenization

C) Byte Pair Encoding (BPE), a type of subword tokenization

D) Byte tokenization

### Answer 6: **C**

### Question 7
What is a primary advantage of subword tokenization over word tokenization?

A) It results in a much smaller vocabulary size.

B) It can represent new or unknown words by breaking them into known subwords.

C) It is significantly faster to train the tokenizer.

D) It preserves the original capitalization of all words perfectly.

### Answer 7: **B**

### Question 8
What is the purpose of a text embedding model like `sentence-transformers/all-mpnet-base-v2`?

A) To generate a unique embedding vector for each token in a sentence.

B) To generate a single embedding vector that represents the meaning of an entire sentence or document.

C) To check for spelling and grammar errors in a text.

D) To compress a text file to a smaller size.

### Answer 8: **B**

### Question 9
In the word2vec algorithm, what is the purpose of the "sliding window"?

A) To determine the size of the embedding vectors.

B) To generate positive training examples of words that appear near each other.

C) To filter out stop words from the text.

D) To visualize the final embeddings in 2D space.

### Answer 9: **B**

### Question 10
When using embeddings for a recommendation system (e.g., for music), what do the "words" and "sentences" correspond to?

A) Words = Artists, Sentences = Albums

B) Words = Songs, Sentences = Playlists

C) Words = Genres, Sentences = Artists

D) Words = Users, Sentences = Songs

### Answer 10: **B**

### Question 11
What does the shape `torch.Size([1, 4, 384])` represent for the output of a contextualized embedding model?

A) 1 sentence, 4 layers, 384-dimensional embeddings.

B) 1 batch, 4 tokens, 384-dimensional embeddings.

C) 1 batch, 4 attention heads, 384 possible next tokens.

D) 1 model, 4 sentences, 384-dimensional embeddings.

### Answer 11: **B**

### Question 12
Why are negative examples crucial for training word2vec?

A) To make the training process faster.

B) To increase the size of the vocabulary.

C) To prevent the model from learning to predict that every word pair is a neighbor.

D) To help the model handle punctuation.

### Answer 12: **C**

### Question 13
Which tokenizer discussed in the chapter is specifically optimized for code and represents individual digits as separate tokens?

A) BERT (uncased)

B) GPT-2

C) GPT-4

D) StarCoder2

### Answer 13: **D**

### Question 14
What is the typical dimensionality of a text embedding from the `all-mpnet-base-v2` model?

A) 50

B) 300

C) 768

D) 4096

### Answer 14: **C**
---

## Part III: Programming Questions (60 points)

### Question 15 (15 points)
**Tokenizer Comparison**

Complete the following Python function to tokenize a given text using two different tokenizers (`bert-base-uncased` and `gpt2`) and compare their outputs.

### Answer 15:

```python
from transformers import AutoTokenizer

def compare_tokenizers(text):
    """
    Tokenizes a text with two different tokenizers and prints the results.
    
    Args:
        text: The string to tokenize.
    """
    tokenizer_names = ["bert-base-uncased", "gpt2"]
    
    for name in tokenizer_names:
        print(f"--- Tokenizer: {name} ---")
        
        # TODO: Load the tokenizer
        tokenizer =  AutoTokenizer.from_pretrained(name) # Your code here
        
        # TODO: Tokenize the text and get the tokens
        tokens = tokenizer(text, return_tensors="pt") # Your code here
        
        # TODO: Convert tokens to IDs
        token_ids = tokens.input_ids.to("cuda") # Your code here
        
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs (first 10): {token_ids[:10]}")
        print("\\n")

# Test your implementation
text_to_tokenize = "Tokenization is a foundational concept in NLP."
compare_tokenizers(text_to_tokenize)
```

The output is:

```
--- Tokenizer: bert-base-uncased ---
Number of tokens: 3
Tokens: {'input_ids': tensor([[  101, 19204,  3989,  2003,  1037,  3192,  2389,  4145,  1999, 17953,
          2361,  1012,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
Token IDs (first 10): tensor([[  101, 19204,  3989,  2003,  1037,  3192,  2389,  4145,  1999, 17953,
          2361,  1012,   102]], device='cuda:0')
\n
--- Tokenizer: gpt2 ---
Number of tokens: 2
Tokens: {'input_ids': tensor([[30642,  1634,   318,   257, 43936,  3721,   287,   399, 19930,    13]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
Token IDs (first 10): tensor([[30642,  1634,   318,   257, 43936,  3721,   287,   399, 19930,    13]],
       device='cuda:0')
\n
```

### Question 16 (15 points)
**Using Pretrained Word2Vec Embeddings**

Complete the following Python script to load a pretrained `word2vec` model from `gensim` and perform similarity operations.

```python
import gensim.downloader as api

def explore_word_embeddings():
    """
    Loads a pretrained word2vec model and explores word similarities.
    """
    # TODO: Load the "glove-wiki-gigaword-50" model
    model = # Your code here
    
    # TODO: Find the 5 most similar words to "woman"
    similar_to_woman = # Your code here
    print("Most similar to 'woman':", similar_to_woman)
    
    # TODO: Find the 5 most similar words to "car"
    similar_to_car = # Your code here
    print("Most similar to 'car':", similar_to_car)
    
    # TODO: Solve the analogy: king - man + woman = ?
    # Find the top 1 result for this analogy.
    analogy_result = # Your code here
    print("Analogy 'king - man + woman':", analogy_result)

# Run the exploration
explore_word_embeddings()
```

### Question 17 (15 points)
**Create your own tokenizer**
```python
from collections import Counter
import re

def create_simple_tokenizer(texts, vocab_size=1000):
    """
    Create a simple BPE-style tokenizer from scratch.
    
    Args:
        texts: List of strings to train the tokenizer on
        vocab_size: Maximum vocabulary size
    
    Returns:
        A dictionary containing the tokenizer vocabulary and encode/decode functions
    """
    
    # TODO: Implement a basic character-level tokenizer that can:
    # 1. Split text into characters initially
    # 2. Count character frequencies
    # 3. Build a vocabulary of the most common characters/subwords
    # 4. Provide encode() and decode() methods
    
    def preprocess_text(text):
        # TODO: Clean and normalize the input text
        # Hint: Convert to lowercase, handle punctuation
        pass
    
    def build_vocab(processed_texts):
        # TODO: Build vocabulary from processed texts
        # Start with character-level tokens, then optionally merge frequent pairs
        pass
    
    def encode(text):
        # TODO: Convert text to token IDs using your vocabulary
        pass
    
    def decode(token_ids):
        # TODO: Convert token IDs back to text
        pass
    
    # Your implementation here
    return {
        'vocab': vocab,
        'encode': encode,
        'decode': decode
    }

# Test your tokenizer
sample_texts = [
    "Hello world! This is a test.",
    "Natural language processing is fascinating.",
    "Tokenization helps models understand text."
]

tokenizer = create_simple_tokenizer(sample_texts, vocab_size=50)
test_text = "Hello! This is new text."
encoded = tokenizer['encode'](test_text)
decoded = tokenizer['decode'](encoded)

print(f"Original: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Vocabulary size: {len(tokenizer['vocab'])}")
```

### Question 18 (15 points)
**Extend the vocabulary of a current tokenizer**
```python
from transformers import AutoTokenizer
import torch

def extend_tokenizer_vocabulary(base_tokenizer_name, new_tokens):
    """
    Extend an existing tokenizer's vocabulary with new tokens.
    
    Args:
        base_tokenizer_name: Name of the base tokenizer (e.g., "bert-base-uncased")
        new_tokens: List of new tokens to add to the vocabulary
    
    Returns:
        Extended tokenizer and demonstration of the new tokens
    """
    
    # TODO: Load the base tokenizer
    tokenizer = # Your code here
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # TODO: Add new tokens to the tokenizer
    # Hint: Use tokenizer.add_tokens() method
    num_added = # Your code here
    
    print(f"Added {num_added} new tokens")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # TODO: Test the extended tokenizer with text containing new tokens
    test_text = "The AI model uses <SPECIAL_TOKEN> for classification."
    
    # Tokenize before and after adding special tokens
    tokens_before = # Your code here (you'll need to reload original tokenizer)
    tokens_after = # Your code here
    
    print(f"\\nTest text: {test_text}")
    print(f"Tokens with original tokenizer: {tokens_before}")
    print(f"Tokens with extended tokenizer: {tokens_after}")
    
    # TODO: Show token IDs for the new tokens
    for token in new_tokens:
        if token in tokenizer.vocab:
            token_id = # Your code here
            print(f"Token '{token}' has ID: {token_id}")
    
    return tokenizer

# Test the function
new_special_tokens = ["<SPECIAL_TOKEN>", "<DOMAIN_TERM>", "<CUSTOM_ENTITY>"]
extended_tokenizer = extend_tokenizer_vocabulary("bert-base-uncased", new_special_tokens)

# Additional test: Show how this affects model input
sample_text = "Process this <SPECIAL_TOKEN> carefully with <DOMAIN_TERM>."
input_ids = extended_tokenizer.encode(sample_text, return_tensors="pt")
print(f"\\nInput IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids}")
```


---

## Submission Guidelines

1. **Format**: Submit your answers in a markdown file named `chapter2_homework_[your_name].md`
2. **Code**: Include all code with proper comments and output.
3. **Explanations**: Provide clear explanations for descriptive questions.

---
