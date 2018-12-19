# Chatbot with Pytorch

## Step 1: Create formatted data file

Used Cornell Movie-Dialogs Corpus as training data set.
Created a formatted data fiel which each line contains a tab-sepearted query sentence and a response sentence pair. 

Sample lines from file:


    b"Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\tWell, I thought we'd start with pronunciation, if that's okay with you.\n"
    b"Well, I thought we'd start with pronunciation, if that's okay with you.\tNot the hacking and gagging and spitting part.  Please.\n"
    b"Not the hacking and gagging and spitting part.  Please.\tOkay... then how 'bout we try out some French cuisine.  Saturday?  Night?\n"
    b"You're asking me out.  That's so cute. What's your name again?\tForget it.\n"
    b"No, no, it's my fault -- we didn't have a proper introduction ---\tCameron.\n"
    b"Cameron.\tThe thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\n"
    b"The thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\tSeems like she could get a date easy enough...\n"


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
The encoder RNN iterates through the input sentence one token (e.g. word) at a time, at each time step outputting an “output” vector and a “hidden state” vector. The hidden state vector is then passed to the next time step, while the output vector is recorded. The encoder transforms the context it saw at each point in the sequence into a set of points in a high-dimensional space, which the decoder will use to generate a meaningful output for the given task.

At the heart of our encoder is a multi-layered Gated Recurrent Unit, invented by Cho et al. in 2014. We will use a bidirectional variant of the GRU, meaning that there are essentially two independent RNNs: one that is fed the input sequence in normal sequential order, and one that is fed the input sequence in reverse order. The outputs of each network are summed at each time step. Using a bidirectional GRU will give us the advantage of encoding both past and future context.

Note that an embedding layer is used to encode our word indices in an arbitrarily sized feature space. For our models, this  layer will map each word to a feature space of size hidden_size. When trained, these values should encode semantic similarity between similar meaning words.

Finally, if passing a padded batch of sequences to an RNN module, we must pack and unpack padding around the RNN pass using  torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence respectively.

Computation Graph:
    1. Convert wor indexes to embeddings
    2. Pack padded batch of sequences for RNN module
    3. Forward pass through GPU
    4. Unpack padding
    5. Sum bidirectional GRU outputs
    6. Return output and final hidden state
    
    
## Step 6: Decoder

The decode RNN generates the response sentence in a token-by-token fashion. It uses the encoder's context vectors, and internal hidden states to generate the next word in the sequence. It continues generating words until it outputs an EOS_token, representing the end of the sentence. A common problem with a vanilla seq2seq decoder is that if we rely soley on the context vector to encode the entire input sequence's meanin, it is likely that we will have information loss. THis is especially the case when dealing with long input sequences, greatly limiting the capability of our decoder. We used the score function.

For the decoddr, we will manually feed our batch one time step at a time.
This means that our-embedded word tensor and GRU output will both have shape (1, batch_size, hidden_size)

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
