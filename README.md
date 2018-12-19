### Chatbot with Pytorch

# Step 1: Create formatted data file

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


# Step 2: Load and trim data

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
 
 
# Step 3: Prepare data for models

Our models expect numerical torch tensors as inputs. We would train our model using mini-batches. # Using mini-batches also means that we must be mindful of the variation of sentence length in our batches. To accomodate sentences of different sizes in the same batch, we will make our batched input tensor of shape  (max_length, batch_size), where sentences shorter than the max_length are zero padded after an EOS_token. If we simply convert our English sentences to tensors by converting words to their indexes(indexesFromSentence) and zero-pad, our tensor would have shape (batch_size, max_length) and indexing the first dimension would return a full sequence across all time-steps. However, we need to be able to index our batch along time, and across all sequences in the batch. Therefore, we transpose our input batch shape to (max_length, batch_size), so that indexing across the first dimension returns a time step across all sentences in the batch. We handle this transpose implicitly in the zeroPadding function.


# Step 4: Define models

We are going to use Seq2Seq(sentence-to-sentence) Model. 
Goal: take a variable-length sequence as an input, and return a variable-length sequence as an output using a fixed-sized model.
One RNN acts as an encoder, which encodes a variable length input sequence to a fixed-length context vector.  In theory, this context vector (the final hidden layer of the RNN) will contain semantic information about the query sentence that is input to the bot. The second RNN is a decoder, which takes an input word and the context vector, and returns a guess for the next word in the sequence and a hidden state to use in the next iteration.


