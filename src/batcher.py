"""This file contains code to process data into batches"""


import time
import random
from threading import Thread
try:
  import queue
except:
  import Queue as queue

import numpy as np
import tensorflow as tf

import data_utils
from model import SummarizationModel


random.seed(123)
FLAGS = tf.app.flags.FLAGS


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, article, abstract_sentences, vocab, hps):
    """Initializes the Example, performing tokenization and truncation to 
    produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      article: source text; a string. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each 
          sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps

    # Get ids of special tokens
    start_decoding = vocab.word2id(data_utils.START_DECODING)
    stop_decoding = vocab.word2id(data_utils.STOP_DECODING)

    # Process the article
    article_words = article.split()
    if len(article_words) > hps.max_enc_steps:
      article_words = article_words[:hps.max_enc_steps]
    self.enc_len = len(article_words) 
    self.enc_input = [vocab.word2id(w) for w in article_words] 

    # Process the abstract
    abstract = ' '.join(abstract_sentences) # string
    abstract_words = abstract.split() 
    abs_ids = [vocab.word2id(w) for w in abstract_words] 

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(
        abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if hps.pointer_gen:
      # Store a version of the enc_input where in-article OOVs are represented 
      # by their temporary OOV id; also store the in-article OOVs words 
      # themselves
      self.enc_input_extend_vocab, self.article_oovs = data_utils.article2ids(
          article_words, vocab)

      # Get a verison of the reference summary where in-article OOVs are 
      # represented by their temporary article OOV id
      abs_ids_extend_vocab = data_utils.abstract2ids(
          abstract_words, vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      # TODO: figure out why dec_input is not changed? Maybe some test?
      _, self.target = self.get_dec_inp_targ_seqs(
          abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_article = article
    self.original_abstract = abstract
    self.original_abstract_sents = abstract_sentences

  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input 
    sequence for the decoder, and the target sequence which we will use to 
    calculate loss. The sequence will be truncated if it is longer than max_len.
    The input sequence must start with the start_id and the target sequence 
    must end with the stop_id (but not if it's been truncated).

    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer

    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there 
        was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target

  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)

  def pad_encoder_input(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if self.hps.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text 
  summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
      example_list: List of Example objects
      hps: hyperparameters
      vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data_utils.PAD_TOKEN) 
    self.init_encoder_seq(example_list, hps) 
    self.init_decoder_seq(example_list, hps) 
    self.store_orig_strings(example_list) 

  def init_encoder_seq(self, example_list, hps):
    """Initializes the encoder sequence
    """
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros(
        (hps.batch_size, max_enc_seq_len), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if hps.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros(
          (hps.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list, hps):
    """Initializes the decoder sequence
    """
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

    # TODO: dynamic_rnn for decoding
    self.dec_batch = np.zeros(
        (hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros(
        (hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros(
        (hps.batch_size, hps.max_dec_steps), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_articles = [ex.original_article for ex in example_list] 
    self.original_abstracts = [ex.original_abstract for ex in example_list] 
    self.original_abstracts_sents = [ex.original_abstract_sents 
        for ex in example_list] 


class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based 
  on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, hps, single_pass, decode_after):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for 
          when you want to run evaluation on the dev or test set). Otherwise 
          generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._single_pass = single_pass
    self._decode_after = decode_after

    # Initialize a queue of Batches waiting to be used, and a queue of Examples 
    # waiting to be batched
    self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, read dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 
      self._finished_reading = False 
    else:
      self._num_example_q_threads = FLAGS.example_queue_threads 
      self._num_batch_q_threads = FLAGS.batch_queue_threads  
      self._bucketing_cache_size = FLAGS.bucketing_cache_size

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if dead
    if not single_pass: # no a watcher in single_pass mode 
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated 
    beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've 
        exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. '
          'Bucket queue size: %i, Input queue size: %i',
          self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed 
    into the example queue."""

    input_gen = self.text_generator(
        data_utils.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (article, abstract) = next(input_gen)
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling "
            "thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading "
              " dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator "
              "is out of data; error.")

      # Use the <s> and </s> tags in abstract to get a list of sentences.
      abstract_sentences = [sent.strip() 
          for sent in data_utils.abstract2sents(abstract)] 
      example = Example(article, abstract_sentences, self._vocab, self._hps) 
      self._example_queue.put(example) 

  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence 
    length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then 
        # sort
        inputs = []
        for _ in range(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len) 

        # Group the sorted Examples into batches, optionally shuffle the 
        # batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          random.shuffle(batches)
        for b in batches:  
          self._batch_queue.put(Batch(b, self._hps, self._vocab))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in range(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._vocab))

  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()

  def text_generator(self, example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See 
        data_utils.example_generator
    """
    cnt = 0
    while True:
      e = next(example_generator) # e is a tf.Example
      try:
        # the article text was saved under the key 'article' in the data files
        article_text = e.features.feature['article'].bytes_list.value[0] 
        # the abstract text was saved under the key 'abstract' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[0] 
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(article_text)==0: 
        # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      else:
        if self._single_pass and cnt < self._decode_after: #skip already decoded docs
          cnt +=1
          continue
        yield (article_text, abstract_text)
