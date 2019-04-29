""" Model helper to build, train, eval, and test model"""


import os
import time
from glob import glob
from collections import namedtuple
from threading import Thread

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bernoulli

import util 
from dqn import DQN
from decode import BeamSearchDecoder
from batcher import Batcher
from data_utils import Vocab
from replay_buffer import ReplayBuffer


FLAGS = tf.app.flags.FLAGS


class ModelHelper(object):
  def __init__(self, model, vocab):
    self.model = model
    self.vocab = vocab

  def calc_running_avg_loss(self, loss, running_avg_loss, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve 
    than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      summary_writer: FileWriter object to write for tensorboard
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is 
          smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
      running_avg_loss = loss
    else:
      running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    self.summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss

  def restore_best_model(self):
    """Load bestmodel file from eval directory, add variables for adagrad, and 
    save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()

  def restore_best_eval_model(self):
    # load best evaluation loss so far
    best_loss = None
    best_step = None
    # goes through all event files and select the best loss achieved and return it
    event_files = sorted(glob('{}/eval/events*'.format(FLAGS.log_root)))
    for ef in event_files:
      try:
        for e in tf.train.summary_iterator(ef):
          for v in e.summary.value:
            step = e.step
            if 'running_avg_loss/decay' in v.tag:
              running_avg_loss = v.simple_value
              if best_loss is None or running_avg_loss < best_loss:
                best_loss = running_avg_loss
                best_step = step
      except:
        continue
    tf.logging.info('resotring best loss from the current logs: {}\tstep: {}'.format(best_loss, best_step))
    return best_loss

  def convert_to_coverage_model(self):
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() 
       if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

  def convert_to_reinforce_model(self):
    """Load non-reinforce checkpoint, add initialized extra variables for 
    reinforce, and save as new checkpoint"""
    tf.logging.info("converting non-reinforce model to reinforce model..")

    # initialize an entire reinforce model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-reinforce weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() 
        if "reinforce" not in v.name and "Adagrad" not in v.name])
    print("restoring non-reinforce variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_rl_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

  def run_training(self, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if FLAGS.ac_training:
      dqn_train_dir = os.path.join(FLAGS.log_root, "dqn", "train")
      if not os.path.exists(dqn_train_dir): os.makedirs(dqn_train_dir)
  
    #replaybuffer_pcl_path = os.path.join(FLAGS.log_root, "replaybuffer.pcl")
    #if not os.path.exists(dqn_target_train_dir): os.makedirs(dqn_target_train_dir)
  
    self.model.build_graph() 
  
    if FLAGS.convert_to_reinforce_model:
      assert (FLAGS.rl_training or FLAGS.ac_training), ('To convert your pointer'
          ' model to a reinforce model, run with convert_to_reinforce_model='
          'True and either rl_training=True or ac_training=True')
      self.convert_to_reinforce_model()
  
    if FLAGS.convert_to_coverage_model:
      assert FLAGS.coverage, ('To convert your non-coverage model to a coverage '
          'model, run with convert_to_coverage_model=True and coverage=True')
      self.convert_to_coverage_model()
  
    if FLAGS.restore_best_model:
      restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
  
    # Loads pre-trained word-embedding. By default the model learns the 
    # embedding.
    if FLAGS.embedding:
      self.vocab.LoadWordEmbedding(FLAGS.embedding, FLAGS.emb_dim)
      word_vector = self.vocab.getWordEmbedding()
  
    # sv = tf.train.Supervisor(logdir=train_dir,
    #     is_chief=True,
    #     saver=saver,
    #     # summary_op=None,
    #     save_summaries_secs=60, 
    #     save_model_secs=60, )
    #     # global_step=model.global_step,
    #     # init_feed_dict={model.embedding_place: word_vector} 
    #     #     if FLAGS.embedding else None)
  
    # self.summary_writer = self.sv.summary_writer
    # self.sess = self.sv.prepare_or_wait_for_session(config=util.get_config())
    self.sess = tf.Session(config=util.get_config())
  
    if FLAGS.ac_training:
      tf.logging.info('DDQN building graph')
      t1 = time.time()
      # We create a separate graph for DDQN
      self.dqn_graph = tf.Graph()
      with self.dqn_graph.as_default():
        self.dqn.build_graph() # build dqn graph
        tf.logging.info('building current network took {} seconds'.format(
            time.time()-t1))
  
        self.dqn_target.build_graph() # build dqn target graph
        tf.logging.info('building target network took {} seconds'.format(
            time.time()-t1))
  
        dqn_saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
        self.dqn_sv = tf.train.Supervisor(logdir=dqn_train_dir,
            is_chief=True,
            saver=dqn_saver,
            summary_op=None,
            save_summaries_secs=60, 
            save_model_secs=60, 
            global_step=self.dqn.global_step)
        self.dqn_summary_writer = self.dqn_sv.summary_writer
        self.dqn_sess = self.dqn_sv.prepare_or_wait_for_session(
            config=util.get_config())
  
      ''' #### TODO: try loading a previously saved replay buffer
      # right now this doesn't work due to running DQN on a thread
      if os.path.exists(replaybuffer_pcl_path):
        tf.logging.info('Loading Replay Buffer...')
        try:
          self.replay_buffer = pickle.load(open(replaybuffer_pcl_path, "rb"))
          tf.logging.info('Replay Buffer loaded...')
        except:
          tf.logging.info('Couldn\'t load Replay Buffer file...')
          self.replay_buffer = ReplayBuffer(self.dqn_hps)
      else:
        self.replay_buffer = ReplayBuffer(self.dqn_hps)
      tf.logging.info("Building DDQN took {} seconds".format(time.time()-t1))
      '''
      self.replay_buffer = ReplayBuffer(self.dqn_hps)
  
    tf.logging.info("Preparing or waiting for session...")
    tf.logging.info("Created session.")
    try:
      self.training_loop(batcher) # this is an infinite loop until interrupted
    except (KeyboardInterrupt, SystemExit):
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      sv.stop()
      if FLAGS.ac_training:
        dqn_sv.stop()

  def training_loop(self, batcher):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    tf.logging.info("Starting run_training")
  
    if FLAGS.debug: # start the tensorflow debugger
      self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
      self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
  
    self.train_step = 0
    # TODO: figure out if this is right
    self.sess.run(tf.global_variables_initializer())

    if FLAGS.ac_training:
      # DDQN training is done asynchronously along with model training
      tf.logging.info('Starting DQN training thread...')
      self.dqn_train_step = 0
      self.thrd_dqn_training = Thread(target=self.dqn_training)
      self.thrd_dqn_training.daemon = True
      self.thrd_dqn_training.start()
  
      watcher = Thread(target=self.watch_threads)
      watcher.daemon = True
      watcher.start()

    # starting the main thread
    tf.logging.info('Starting Seq2Seq training...')
    while True: # repeats until interrupted
      batch = batcher.next_batch()
      t0=time.time()
      if FLAGS.ac_training:
        # For DDQN:
        # 1) collect model output to calculate reward and Q-estimates
        # 2）fix estimation either using target network or using true Q-values
        # This process will usually take time. We are working on improving it.

        # len(batch_size * k * max_dec_steps)
        transitions = self.model.collect_dqn_transitions(
            self.sess, batch, self.train_step, batch.max_art_oovs) 
        tf.logging.info('Q-values collection time: {}'.format(time.time()-t0))
        # whenever we are working with the DDQN, we switch using DDQN graph 
        # rather than default graph
        with self.dqn_graph.as_default():
          batch_len = len(transitions)
          # we use current decoder state to predict q_estimates, 
          # use_state_prime = False
          b = ReplayBuffer.create_batch(
              self.dqn_hps, 
              transitions,
              len(transitions), 
              use_state_prime=False, 
              max_art_oovs=batch.max_art_oovs)
          # we also get the next decoder state to correct the estimation, 
          # use_state_prime = True
          b_prime = ReplayBuffer.create_batch(
              self.dqn_hps, 
              transitions,
              len(transitions), 
              use_state_prime = True, 
              max_art_oovs=batch.max_art_oovs)
          # use current DQN to estimate values from current decoder state
          dqn_results = self.dqn.run_test_steps(
              sess=self.dqn_sess, x= b._x, return_best_action=True)
          # q_estimates shape (len(transitions), vocab_size)
          q_estimates = dqn_results['estimates'] 
          dqn_best_action = dqn_results['best_action']
          #dqn_q_estimate_loss = dqn_results['loss']
  
          # use target DQN to estimate values for the next decoder state
          dqn_target_results = self.dqn_target.run_test_steps(
              self.dqn_sess, x= b_prime._x)
          # q_vals_net_t (len(transitions), vocab_size)
          q_vals_new_t = dqn_target_results['estimates'] 
  
          # Expand the q_estimates to match the input batch max_art_oov
          # Use the q_estimate of UNK token for all the OOV tokens
          q_estimates = np.concatenate([q_estimates,
              np.reshape(q_estimates[:,0],[-1,1])*np.ones((len(transitions),
                         batch.max_art_oovs))], axis=-1)
          # modify Q-estimates using the result collected from current and 
          # target DQN. check algorithm 5 in the paper for more info: 
          # https://arxiv.org/pdf/1805.09461.pdf

          for i, tr in enumerate(transitions):
            if tr.done:
              q_estimates[i][tr.action] = tr.reward
            else:
              q_estimates[i][tr.action] = tr.reward + \
                  FLAGS.gamma * q_vals_new_t[i][dqn_best_action[i]]

          # use scheduled sampling to whether use true Q-values or DDQN 
          # estimation
          if FLAGS.dqn_scheduled_sampling:
            q_estimates = self.scheduled_sampling(
                batch_len, 
                FLAGS.sampling_probability, 
                b._y_extended, 
                q_estimates)

          if not FLAGS.calculate_true_q:
            # when we are not training DDQN based on true Q-values,
            # we need to update Q-values in our transitions based on the 
            # q_estimates we collected from DQN current network.
            for trans, q_val in zip(transitions,q_estimates):
              trans.q_values = q_val # each have the size vocab_extended

          # shape (batch_size, k, max_dec_steps, vocab_size_extended)
          q_estimates = np.reshape(
              q_estimates, 
              [FLAGS.batch_size, FLAGS.k, FLAGS.max_dec_steps, -1]) 

        # Once we are done with modifying Q-values, we can use them to train 
        # the DDQN model. In this paper, we use a priority experience buffer 
        # which always selects states with higher quality to train the DDQN. 
        # The following line will add batch_size * max_dec_steps experiences to
        # the replay buffer.
        # As mentioned before, the DDQN training is asynchronous. Therefore, 
        # once the related queues for DDQN training are full, the DDQN will 
        # start the training.
        self.replay_buffer.add(transitions)
        # If dqn_pretrain flag is on, it means that we use a fixed Actor to only
        # collect experiences for DDQN pre-training
        if FLAGS.dqn_pretrain:
          tf.logging.info(
              'RUNNNING DQN PRETRAIN: Adding data to relplay buffer only...')
          continue

        # if not, use the q_estimation to update the loss.
        results = self.model.run_train_steps(
            self.sess, batch, self.train_step, q_estimates)
      else:
        results = self.model.run_train_steps(self.sess, batch, self.train_step)
      t1=time.time()

      # get summaries and iteration number 
      summaries = results['summaries'] 
      self.train_step = results['global_step'] 
      tf.logging.info('seconds for training step {}: {}'.format(
          self.train_step, t1-t0))
  
      printer_helper = {}
      printer_helper['pgen_loss']= results['pgen_loss']
      if FLAGS.coverage:
        printer_helper['coverage_loss'] = results['coverage_loss']
        if FLAGS.rl_training or FLAGS.ac_training:
          printer_helper['rl_cov_total_loss'] = results['reinforce_cov_total_loss']
        else:
          printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
      if FLAGS.rl_training or FLAGS.ac_training:
        printer_helper['shared_loss'] = results['shared_loss']
        printer_helper['rl_loss'] = results['rl_loss']
        printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
      if FLAGS.rl_training:
        printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
        printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
        printer_helper['r_diff'] = printer_helper['greedy_r'] - printer_helper['sampled_r']
      if FLAGS.ac_training:
        if len(self.avg_dqn_loss)>0:
          printer_helper['dqn_loss'] = np.mean(self.avg_dqn_loss) 
        else:
          printer_helper['dqn_loss'] = 0
  
      for (k,v) in printer_helper.items():
        if not np.isfinite(v):
          raise Exception("{} is not finite. Stopping.".format(k))
        tf.logging.info('{}: {}\t'.format(k,v))
      tf.logging.info('-------------------------------------------')
  
      self.summary_writer.add_summary(summaries, self.train_step) 
      if self.train_step % 100 == 0: # flush the summary writer every so often
        self.summary_writer.flush()
        if FLAGS.ac_training:
          self.dqn_summary_writer.flush()

      if self.train_step > FLAGS.max_iter: 
        break
  
  def dqn_training(self):
    """ training the DDQN network."""
    try:
      while True:
        if self.dqn_train_step == FLAGS.dqn_pretrain_steps: raise SystemExit()
        _t = time.time()
        self.avg_dqn_loss = []
        avg_dqn_target_loss = []
        # Get a batch of size dqn_batch_size from replay buffer to train the model
        dqn_batch = self.replay_buffer.next_batch()
        if dqn_batch is None:
          tf.logging.info('replay buffer not loaded enough yet...')
          time.sleep(60)
          continue
        # Run train step for Current DQN model and collect the results
        dqn_results = self.dqn.run_train_steps(self.dqn_sess, dqn_batch)
        # Run test step for Target DQN model and collect the results and 
        # monitor the difference in loss between the two
        dqn_target_results = self.dqn_target.run_test_steps(
            self.dqn_sess, x=dqn_batch._x, y=dqn_batch._y, return_loss=True)
        self.dqn_train_step = dqn_results['global_step']
        self.dqn_summary_writer.add_summary(
            dqn_results['summaries'], self.dqn_train_step) # write summaries
        self.avg_dqn_loss.append(dqn_results['loss'])
        avg_dqn_target_loss.append(dqn_target_results['loss'])
        self.dqn_train_step = self.dqn_train_step + 1
        tf.logging.info('seconds for training dqn model: {}'.format(time.time()-_t))
        # UPDATING TARGET DDQN NETWORK WITH CURRENT MODEL
        with self.dqn_graph.as_default():
          current_model_weights = self.dqn_sess.run(
              [self.dqn.model_trainables])[0] # get weights of current model
          self.dqn_target.run_update_weights(
              self.dqn_sess, self.dqn_train_step, current_model_weights) 

        tf.logging.info('DQN loss at step {}: {}'.format(
            self.dqn_train_step, np.mean(self.avg_dqn_loss)))
        tf.logging.info('DQN Target loss at step {}: {}'.format(
            self.dqn_train_step, np.mean(avg_dqn_target_loss)))
        # sleeping is required if you want the keyboard interuption to work
        time.sleep(FLAGS.dqn_sleep_time)
    except (KeyboardInterrupt, SystemExit):
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      self.sv.stop()
      self.dqn_sv.stop()
  
  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      if not self.thrd_dqn_training.is_alive(): # if the thread is dead
        tf.logging.error('Found DQN Learning thread dead. Restarting.')
        self.thrd_dqn_training = Thread(target=self.dqn_training)
        self.thrd_dqn_training.daemon = True
        self.thrd_dqn_training.start()
  
  def run_eval(self, batcher):
    """Repeatedly runs eval iterations, logging to screen and writing summaries.
    Saves the model with the best loss seen so far."""
    self.model.build_graph() 
    saver = tf.train.Saver(max_to_keep=3) 
    sess = tf.Session(config=util.get_config())
  
    if FLAGS.embedding:
      sess.run(tf.global_variables_initializer(),
          feed_dict={self.model.embedding_place:self.word_vector})
    eval_dir = os.path.join(FLAGS.log_root, "eval") 
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') 
    self.summary_writer = tf.summary.FileWriter(eval_dir)
  
    if FLAGS.ac_training:
      tf.logging.info('DDQN building graph')
      t1 = time.time()
      dqn_graph = tf.Graph()
      with dqn_graph.as_default():
        self.dqn.build_graph() # build dqn graph
        tf.logging.info('building current network took {} seconds'.format(
            time.time()-t1))
        self.dqn_target.build_graph() # build dqn target graph
        tf.logging.info('building target network took {} seconds'.format(
            time.time()-t1))
        dqn_saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
        dqn_sess = tf.Session(config=util.get_config())
      dqn_train_step = 0
      replay_buffer = ReplayBuffer(self.dqn_hps)

    # the eval job keeps a smoother, running average loss to tell it when to 
    # implement early stopping
    running_avg_loss = 0 
    best_loss = self.restore_best_eval_model()  
    train_step = 0
  
    while True:
      _ = util.load_ckpt(saver, sess) # load a new checkpoint
      if FLAGS.ac_training:
        _ = util.load_dqn_ckpt(dqn_saver, dqn_sess) # load a new checkpoint
      processed_batch = 0
      avg_losses = []
      # evaluate for 100 * batch_size before comparing the loss
      # we do this due to memory constraint, best to run eval on different 
      # machines with large batch size
      while processed_batch < 100*FLAGS.batch_size:
        processed_batch += FLAGS.batch_size
        batch = batcher.next_batch() # get the next batch
        if FLAGS.ac_training:
          t0 = time.time()
          transitions = self.model.collect_dqn_transitions(
              sess, batch, train_step, batch.max_art_oovs) 
          tf.logging.info('Q values collection time: {}'.format(time.time()-t0))
          with dqn_graph.as_default():
            # if using true Q-value to train DQN network,
            # we do this as the pre-training for the DQN network to get better
            # estimates
            batch_len = len(transitions)
            b = ReplayBuffer.create_batch(
                self.dqn_hps, 
                transitions,
                len(transitions), 
                use_state_prime=True, 
                max_art_oovs=batch.max_art_oovs)
            b_prime = ReplayBuffer.create_batch(
                self.dqn_hps, 
                transitions,
                len(transitions), 
                use_state_prime=True, 
                max_art_oovs=batch.max_art_oovs)
            dqn_results = self.dqn.run_test_steps(
                sess=dqn_sess, x= b._x, return_best_action=True)
            # q_estimates shape (len(transitions), vocab_size)
            q_estimates = dqn_results['estimates'] 
            dqn_best_action = dqn_results['best_action']
  
            tf.logging.info('running test step on dqn_target')
            dqn_target_results = self.dqn_target.run_test_steps(
                dqn_sess, x= b_prime._x)
            # q_vals_new_t shape (len(transitions), vocab_size)
            q_vals_new_t = dqn_target_results['estimates'] 
  
            # we need to expand the q_estimates to match the input 
            # batch max_art_oov
            q_estimates = np.concatenate(
                [q_estimates, np.zeros((len(transitions),batch.max_art_oovs))],
                axis=-1)
  
            tf.logging.info('fixing the action q-estimates')
            for i, tr in enumerate(transitions):
              if tr.done:
                q_estimates[i][tr.action] = tr.reward
              else:
                q_estimates[i][tr.action] = tr.reward + \
                    FLAGS.gamma * q_vals_new_t[i][dqn_best_action[i]]
            if FLAGS.dqn_scheduled_sampling:
              tf.logging.info('scheduled sampling on q-estimates')
              q_estimates = self.scheduled_sampling(
                  batch_len, 
                  FLAGS.sampling_probability, 
                  b._y_extended, 
                  q_estimates)
            if not FLAGS.calculate_true_q:
              # when we are not training DQN based on true Q-values
              # we need to update Q-values in our transitions based on this 
              # q_estimates we collected from DQN current network.
              for trans, q_val in zip(transitions,q_estimates):
                trans.q_values = q_val # each have the size vocab_extended
            # q_estimates (batch_size, k, max_dec_steps, vocab_size_extended)
            q_estimates = np.reshape(
                q_estimates, 
                [FLAGS.batch_size, FLAGS.k, FLAGS.max_dec_steps, -1]) 

          tf.logging.info('run eval step on seq2seq model.')
          t0=time.time()
          results = self.model.run_eval_step(
              sess, batch, train_step, q_estimates)
          t1=time.time()
        else:
          tf.logging.info('run eval step on seq2seq model.')
          t0=time.time()
          results = self.model.run_eval_step(sess, batch, train_step)
          t1=time.time()
  
        tf.logging.info('experiment: {}'.format(FLAGS.exp_name))
        tf.logging.info('processed_batch: {}, seconds for batch: {}'.format(
            processed_batch, t1-t0))
  
        printer_helper = {}
        loss = printer_helper['pgen_loss']= results['pgen_loss']
        if FLAGS.coverage:
          printer_helper['coverage_loss'] = results['coverage_loss']
          if FLAGS.rl_training or FLAGS.ac_training:
            printer_helper['rl_cov_total_loss']= results['reinforce_cov_total_loss']
          loss = printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
        if FLAGS.rl_training or FLAGS.ac_training:
          printer_helper['shared_loss'] = results['shared_loss']
          printer_helper['rl_loss'] = results['rl_loss']
          printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
        if FLAGS.rl_training:
          printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
          printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
          printer_helper['r_diff'] = printer_helper['greedy_r'] - printer_helper['sampled_r']
        if FLAGS.ac_training:
          if len(self.avg_dqn_loss) > 0:
            printer_helper['dqn_loss'] = np.mean(self.avg_dqn_loss)
          else: 
            printer_helper['dqn_loss'] = 0
  
        for (k,v) in printer_helper.items():
          if not np.isfinite(v):
            raise Exception("{} is not finite. Stopping.".format(k))
          tf.logging.info('{}: {}\t'.format(k,v))
  
        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        self.summary_writer.add_summary(summaries, train_step)
  
        # calculate running avg loss
        avg_losses.append(self.calc_running_avg_loss(np.asscalar(loss), 
                          running_avg_loss, train_step))
        tf.logging.info('-------------------------------------------')
  
      running_avg_loss = np.mean(avg_losses)
      tf.logging.info('==========================================')
      tf.logging.info('best_loss: {}\trunning_avg_loss: {}\t'.format(
          best_loss, running_avg_loss))
      tf.logging.info('==========================================')
  
      # If running_avg_loss is best so far, save this checkpoint (early 
      # stopping). These checkpoints will appear as 
      # bestmodel-<iteration_number> in the eval dir
      if best_loss is None or running_avg_loss < best_loss:
        tf.logging.info('Found new best model with %.3f running_avg_loss. '
            'Saving to %s', running_avg_loss, bestmodel_save_path)
        saver.save(
            sess, 
            bestmodel_save_path, 
            global_step=train_step, 
            latest_filename='checkpoint_best')
        best_loss = running_avg_loss
  
      # flush the summary writer every so often
      if train_step % 100 == 0:
        self.summary_writer.flush()
      #time.sleep(600) # run eval every 10 minute
  
  # Scheduled sampling used for either selecting true Q-estimates or the 
  # DDQN estimation based on 
  # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/ScheduledEmbeddingTrainingHelper
  def scheduled_sampling(self, batch_size, sampling_probability, true, estimate):
    with variable_scope.variable_scope("ScheduledEmbedding"):
      # Return -1s where we do not sample, and sample_ids elsewhere
      select_sampler = bernoulli.Bernoulli(
          probs=sampling_probability, dtype=tf.bool)
      select_sample = select_sampler.sample(sample_shape=batch_size)
      sample_ids = array_ops.where(
          select_sample,
          tf.range(batch_size),
          gen_array_ops.fill([batch_size], -1))
      where_sampling = math_ops.cast(
          array_ops.where(sample_ids > -1), tf.int32)
      where_not_sampling = math_ops.cast(
          array_ops.where(sample_ids <= -1), tf.int32)
      _estimate = array_ops.gather_nd(estimate, where_sampling)
      _true = array_ops.gather_nd(true, where_not_sampling)
  
      base_shape = array_ops.shape(true)
      result1 = array_ops.scatter_nd(
          indices=where_sampling, updates=_estimate, shape=base_shape)
      result2 = array_ops.scatter_nd(
          indices=where_not_sampling, updates=_true, shape=base_shape)
      result = result1 + result2
      return result1 + result2

