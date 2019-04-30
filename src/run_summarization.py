"""This is the top-level file to train, evaluate or test your summarization 
model"""


import os
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

from data_utils import Vocab
from batcher import Batcher
from dqn import DQN
from model import SummarizationModel
from model_helper import ModelHelper


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
tf.set_random_seed(111) # a seed value for randomness

# Where to find data
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles.'
    ' Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('vocab_path', '', 'Path expression to text vocab file.')

# Important settings
flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run'
    'eval on the full dataset using a fixed checkpoint, i.e. take the current '
    'checkpoint, and use it to produce one summary for each example in the '
    'dataset, write the summaries to file and then get ROUGE scores for the '
    'whole dataset. If False (default), run concurrent decoding, i.e. '
    'repeatedly load latest checkpoint, use it to produce summaries for '
    'randomly-chosen examples and log the results to screen, indefinitely.')
flags.DEFINE_integer('decode_after', 0, 'skip already decoded docs')
flags.DEFINE_string('decode_from', 'train', 'Decode from train/eval model.')

# Where to save output
flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved '
    'in a directory with this name, under log_root.')

# batcher parameter, for consistent results, set all these parameters to 1
flags.DEFINE_integer('example_queue_threads', 4, 'Number of example queue '
    'threads,')
flags.DEFINE_integer('batch_queue_threads', 2, 'Number of batch queue threads.')
flags.DEFINE_integer('bucketing_cache_size', 100, 'Number of bucketing cache '
    'size.')

# Hyperparameters
flags.DEFINE_string('embedding', None, 'path to the pre-trained embedding file')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
flags.DEFINE_integer('enc_hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('dec_hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('batch_size', 64, 'minibatch size')
flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max '
    'source text tokens)')
flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max '
    'summary tokens)')
flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of '
    'generated summary. Applies only for beam search decoding mode')
flags.DEFINE_integer('max_iter', 55000, 'max number of iterations')
flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be '
    'read from the vocabulary file in order. If the vocabulary file contains '
    'fewer words than this number, or if this number is set to 0, will take '
    'all words in the vocabulary file.')
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for '
    'Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells '
    'random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used '
    'for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
flags.DEFINE_integer('gpu_num', 0, 'which gpu to use to train the model')

# Pointer-generator or baseline model
flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator '
    'model. If False, use baseline model.')
flags.DEFINE_boolean('avoid_trigrams', True, 'Avoids trigram during decoding')
# Eq 13. in https://arxiv.org/pdf/1705.04304.pdf
flags.DEFINE_boolean('share_decoder_weights', False, 'Share output matrix '
    'projection with word embedding') 

# Pointer-generator with Self-Critic policy gradient:
# https://arxiv.org/pdf/1705.04304.pdf
flags.DEFINE_boolean('rl_training', False, 'Use policy-gradient training by '
    'collecting rewards at the end of sequence.')
flags.DEFINE_boolean('self_critic', True, 'Uses greedy sentence reward as '
    'baseline.')
flags.DEFINE_boolean('use_discounted_rewards', False, 'Whether to use '
    'discounted rewards.')
flags.DEFINE_boolean('use_intermediate_rewards', False, 'Whether to use '
    'intermediate rewards.')
flags.DEFINE_boolean('convert_to_reinforce_model', False, 'Convert a pointer '
    'model to a reinforce model. Turn this on and run in train mode. Your '
    'current training model will be copied to a new version (same name with '
    '_cov_init appended) that will be ready to run with coverage flag turned '
    'on, for the coverage training stage.')
flags.DEFINE_boolean('intradecoder', False, 'Use intradecoder attention or not')
flags.DEFINE_boolean('use_temporal_attention', False, 'Whether to use temporal '
    'attention or not')
flags.DEFINE_boolean('matrix_attention', False, 'Use matrix attention, Eq. 2 '
    'https://arxiv.org/pdf/1705.04304.pdf')
flags.DEFINE_float('eta', 0, 'RL/MLE scaling factor, 1 means use RL loss, 0 '
    'means use MLE loss')
flags.DEFINE_boolean('fixed_eta', False, 'Use fixed value for eta or adaptive '
    'based on global step')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_string('reward_function', 'rouge_l/f_score', 'either bleu or one '
    'of the rouge measures (rouge_1/f_score,rouge_2/f_score,rouge_l/f_score)')

# parameters of DDQN model
flags.DEFINE_boolean('ac_training', False, 'Use Actor-Critic learning by DDQN.')
flags.DEFINE_boolean('dqn_scheduled_sampling', False, 'Whether to use '
    'scheduled sampling to use estimates of dqn model vs the actual q-estimates'
    ' values')
flags.DEFINE_string('dqn_layers', '512,256,128', 'DQN dense hidden layer size, '
    'will create three dense layers with 512, 256, and 128 size')
flags.DEFINE_integer('dqn_replay_buffer_size', 100000, 'Size of the replay '
    'buffer')
flags.DEFINE_integer('dqn_batch_size', 100, 'Batch size for training the DDQN '
    'model')
flags.DEFINE_integer('dqn_target_update', 10000, 'Update target Q network '
    'every 10000 steps')
flags.DEFINE_integer('dqn_sleep_time', 2, 'Train DDQN model every 2 seconds')
flags.DEFINE_integer('dqn_gpu_num', 0, 'GPU number to train the DDQN')
flags.DEFINE_boolean('dueling_net', True, 'Whether to use Duelling Network to '
    'train the model') # https://arxiv.org/pdf/1511.06581.pdf
flags.DEFINE_boolean('dqn_polyak_averaging', True, 'Whether to use polyak '
    'averaging to update the target network parameters')
flags.DEFINE_boolean('calculate_true_q', False, 'Whether to use true Q-values '
    'to train DQN or use DQN\'s estimates to train it')
flags.DEFINE_boolean('dqn_pretrain', False, 'Pretrain the DDQN network with '
    'fixed Actor model')
flags.DEFINE_integer('dqn_pretrain_steps', 10000, 'Number of steps to '
    'pre-train the DDQN')

#scheduled sampling parameters, https://arxiv.org/pdf/1506.03099.pdf
# At each time step t and for each sequence in the batch, we get the input to 
# next decoding step by either
#   (1) sampling from the final distribution at (t-1), or
#   (2) reading from input_decoder_embedding.
# We do (1) with probability sampling_probability and (2) with 
# 1 - sampling_probability.
# Using sampling_probability=0.0 is equivalent to using only the ground truth
# data (no sampling).
# Using sampling_probability=1.0 is equivalent to doing inference by only 
# relying on the sampled token generated at each decoding step
flags.DEFINE_boolean('scheduled_sampling', False, 'whether to do scheduled '
    'sampling or not')
#### TODO: implement this 
flags.DEFINE_string('decay_function', 'linear','linear, exponential, inv_sigmoid') 
flags.DEFINE_float('sampling_probability', 0, 'epsilon value for choosing '
    'ground-truth or model output')
flags.DEFINE_boolean('fixed_sampling_probability', False, 'Whether to usefixed '
    'sampling probability or adaptive based on global step')
flags.DEFINE_boolean('hard_argmax', True, 'Whether to use soft argmax or hard '
    'argmax')
flags.DEFINE_boolean('greedy_scheduled_sampling', False, 'Whether to use '
    'greedy approach or sample for the output, if True it uses greedy')
flags.DEFINE_boolean('E2EBackProp', False, 'Whether to use E2EBackProp '
    'algorithm to solve exposure bias')
flags.DEFINE_float('alpha', 1, 'soft argmax argument')
flags.DEFINE_integer('k', 1, 'number of samples')
flags.DEFINE_boolean('scheduled_sampling_final_dist', True, 'Whether to use '
    'final distribution or vocab distribution for scheduled sampling')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the '
    'experiments reported in the ACL paper train WITHOUT coverage until '
    'converged, and then train for a short phase WITH coverage afterwards. i.e.'
    ' to reproduce the results in the ACL paper, turn this off for most of '
    'training then turn on for a short phase at the end.')
flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the '
    'paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a '
    'non-coverage model to a coverage model. Turn this on and run in train '
    'mode. Your current training model will be copied to a new version (same '
    'name with _cov_init appended) that will be ready to run with coverage '
    'flag turned on, for the coverage training stage.')
flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in '
    'the eval/ dir and save it in the train/ dir, ready to be used for further'
    ' training. Useful for early stopping, or if your training checkpoint has '
    'become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
flags.DEFINE_boolean('debug', False, 'Run in tensorflow\'s debug mode (watches '
    'for NaN/inf values)')


def set_log():
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  tf.logging.set_verbosity(tf.logging.INFO) 
  tf.logging.info('Starting seq2seq in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if 
  # necessary
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode == "train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it."
          % (FLAGS.log_root))


def save_config():
  fw = open('{}/config.txt'.format(FLAGS.log_root), 'w')
  flags = getattr(FLAGS, "__flags")
  for k, v in flags.items():
    fw.write('{}\t{}\n'.format(k, v))
  fw.close()


def get_hparams():
  """Get hyperparameters"""
  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to
  # make a batch of these hypotheses.
  if FLAGS.mode == 'decode':
    FLAGS.batch_size = FLAGS.beam_size

  # If single_pass=True, check we're in decode mode
  if FLAGS.single_pass and FLAGS.mode != 'decode':
    raise Exception("The single_pass flag should only be True in decode mode")

  # Make a namedtuple hps, containing the values of the hyperparameters that 
  # the model needs
  hparam_list = [
      'mode', 
      'lr', 
      'gpu_num',
      #'sampled_greedy_flag', 
      'gamma', 
      'eta', 
      'fixed_eta', 
      'reward_function', 
      'intradecoder', 
      'use_temporal_attention', 
      'ac_training',
      'rl_training', 
      'matrix_attention', 
      'calculate_true_q',
      'enc_hidden_dim', 
      'dec_hidden_dim', 'k', 
      'scheduled_sampling', 
      'sampling_probability',
      'fixed_sampling_probability',
      'alpha', 
      'hard_argmax', 
      'greedy_scheduled_sampling',
      'adagrad_init_acc', 
      'rand_unif_init_mag', 
      'trunc_norm_init_std', 
      'max_grad_norm', 
      'emb_dim', 'batch_size', 
      'max_dec_steps', 
      'max_enc_steps',
      'dqn_scheduled_sampling', 
      'dqn_sleep_time', 
      'E2EBackProp',
      'coverage', 
      'cov_loss_wt',
      'pointer_gen']

  hps_dict = {}
  for key,val in FLAGS.flag_values_dict().items():
    if key in hparam_list: 
      hps_dict[key] = val
  if FLAGS.ac_training:
    hps_dict.update({'dqn_input_feature_len': (FLAGS.dec_hidden_dim)})
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
  return hps


def dqn_hparams():
  """creating all the required parameters for DDQN model."""
  hparam_list = [
      'lr', 
      'dqn_gpu_num', 
      'dqn_layers', 
      'dqn_replay_buffer_size', 
      'dqn_batch_size', 
      'dqn_target_update',
      'dueling_net',
      'dqn_polyak_averaging',
      'dqn_sleep_time',
      'dqn_scheduled_sampling',
      'max_grad_norm']
  hps_dict = {}
  for key,val in FLAGS.flag_values_dict().items():
    if key in hparam_list: 
      hps_dict[key] = val
  hps_dict.update({'dqn_input_feature_len':(FLAGS.dec_hidden_dim)})
  hps_dict.update({'vocab_size':self.vocab.size()})
  dqn_hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
  return dqn_hps


def main(unused_argv):
  if len(unused_argv) != 1: 
    raise Exception("Problem with flags: %s" % unused_argv)

  # setting
  set_log()
  save_config()

  # hyperparameters
  hps = get_hparams()
  if FLAGS.ac_training:
    dqn_hps = get_dqn_hparams()

  # load vocab
  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) 
  # Create a batcher object that will create minibatches of data
  batcher = Batcher(FLAGS.data_path, 
      vocab, 
      hps, 
      single_pass=FLAGS.single_pass, 
      decode_after=FLAGS.decode_after)


  if hps.mode == 'train':
    print("creating model...")
    model = SummarizationModel(hps, vocab)
    helper = ModelHelper(model, vocab)
    if FLAGS.ac_training:
      # current DQN with paramters \Psi
      helper.dqn = DQN(self.dqn_hps,'current')
      # target DQN with paramters \Psi^{\prime}
      helper.dqn_target = DQN(self.dqn_hps,'target')
    helper.run_training(batcher)

  elif hps.mode == 'eval':
    model = SummarizationModel(hps, vocab)
    helper = ModelHelper(model, vocab)
    if FLAGS.ac_training:
      helper.dqn = DQN(dqn_hps,'current')
      helper.dqn_target = DQN(dqn_hps,'target')
    helper.run_eval(batcher)

  elif hps.mode == 'decode':
    # This will be the hyperparameters for the decoder model
    decode_model_hps = hps  
    # The model is configured with max_dec_steps=1 because we only ever run one
    # step of the decoder at a time (to do beam search). Note that the batcher 
    # is initialized with max_dec_steps equal to e.g. 100 because the batches 
    # need to contain the full summaries
    decode_model_hps = hps._replace(max_dec_steps=1) 
    model = SummarizationModel(decode_model_hps, self.vocab)
    if FLAGS.ac_training:
      # We need our target DDQN network for collecting Q-estimation at each 
      # decoder step.
      dqn_target = DQN(self.dqn_hps,'target')
    else:
      dqn_target = None
    decoder = BeamSearchDecoder(model, self.batcher, self.vocab, dqn=dqn_target)
    # decode indefinitely (unless single_pass=True, in which case deocde the 
    # dataset exactly once)
    decoder.decode() 
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
  tf.app.run()
