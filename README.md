# Chatbot with Pytorch

## Step 1: Create formatted data file

Used Cornell Movie-Dialogs Corpus as training data set.
Created a formatted data fiel which each line contains a tab-sepearted query sentence and a response sentence pair. 


## Step 2: Load and trim data

Created a vocabulary and load query/response sentence pairs into memory.
We are dealing with sequences of words, which do not have an implicit mapping to a discrete numerical space. Thus, we must create one by mapping each unique word that we encounter in our dataset to an index value.
For this step, we create a new class, "voc" which keeps a mapping from words to indexes, a reverse mapping of indexed to words, a count of each word and a total word count. 
We can assemble our vocabulary and query/respose sentence pairs. Before we are ready to use this data, we must perform some preprocessing. 
  - Convert the Unicode strings to ASCII.
  - Convert all letters to lowercase and trim all non-letter characters except for basic puncutuation.
  - To aid in training convergence, we filter out sentences with length greater than the MAX_LENGTH threshold
Another tactic that is benefitial to achieving faster convergence during training is trimming rearely used words out of our voacbulary. Decreasing the feature space will also soften the difficult of the function that the model must learn to approximate.
We will do this as a two-step process:
  - Trim words used under MIN_COUNT threshold using hte voc.trim function
  - Filter out pairs with trimmed words
 
 
## Step 3: Prepare data for models

Our models expect numerical torch tensors as inputs. We would train our model using mini-batches. # Using mini-batches also means that we must be mindful of the variation of sentence length in our batches. To accomodate sentences of different sizes in the same batch, we will make our batched input tensor of shape  (max_length, batch_size), where sentences shorter than the max_length are zero padded after an EOS_token. If we simply convert our English sentences to tensors by converting words to their indexes(indexesFromSentence) and zero-pad, our tensor would have shape (batch_size, max_length) and indexing the first dimension would return a full sequence across all time-steps. However, we need to be able to index our batch along time, and across all sequences in the batch. Therefore, we transpose our input batch shape to (max_length, batch_size), so that indexing across the first dimension returns a time step across all sentences in the batch. We handle this transpose implicitly in the zeroPadding function.


## Step 4: Define models

We are going to use Seq2Seq(sentence-to-sentence) Model. 
Goal: take a variable-length sequence as an input, and return a variable-length sequence as an output using a fixed-sized model.
One RNN acts as an encoder, which encodes a variable length input sequence to a fixed-length context vector.  In theory, this context vector (the final hidden layer of the RNN) will contain semantic information about the query sentence that is input to the bot. The second RNN is a decoder, which takes an input word and the context vector, and returns a guess for the next word in the sequence and a hidden state to use in the next iteration.


## Step 5: Encoder

Computation Graph:
  1. Convert wor indexes to embeddings
  2. Pack padded batch of sequences for RNN module
  3. Forward pass through GPU
  4. Unpack padding
  5. Sum bidirectional GRU outputs
  6. Return output and final hidden state
    
    
## Step 6: Decoder

Computation Graph:
  1. Get embedding of current input word
  2. Forward through unidirectional GRU
  3. Calculate attention weights from the current GRU output from (2)
  4. Multiply attention weights to encoder outputs to get new "weighted sum" context vector
  5. Concatenate weighted context vector and GRU output using Luong equation 5
  6. Predict next word using Luong equation 6 (without softmax)
  7. Return output and final hidden state
    
    
## Step 7: Single training

Sequence of Operations:
  1. Forward pass entire input batch through encoder.
  2. Initialize decoder inputs as SOS_token, and hidden state as the encoder’s final hidden state.
  3. Forward input batch sequence through decoder one time step at a time.
  4. If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
  5. Calculate and accumulate loss.
  6. Perform backpropagation.
  7. Clip gradients.
  8. Update encoder and decoder model parameters.


## Step 8: Define evaluation

Greedy decoding

Computation graph:

   1. Forward input through encoder model.
   2. Prepare encoder's final hidden layer to be first hidden input to the decoder.
   3. Initialize decoder's first input as SOS_token.
   4. Initialize tensors to append decoded words to.
   5. Iteratively decode one word token at a time:
       a) Forward pass through decoder.
       b) Obtain most likely word token and its softmax score.
       c) Record token and score.
       d) Prepare current token to be next decoder input.
   6. Return collections of word tokens and scores.


## Funny Demo:

![alt text](https://github.com/billyshin/Chatbot_Pytorch/blob/master/Screen-Shot-2019-01-05-at-14.45.21-PM.png)
