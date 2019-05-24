#!/usr/bin/env python3

import json
import os

import fire
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

import encoder
import model
import sample
from encoder import Encoder


def run_model(
        starting_text: str,
        model_name='345M',
        seed=None,
        nsamples=1,
        length=None,
        temperature=1,
        top_k=0,
        models_dir='models',
):
    if starting_text.strip() == '':
        raise ValueError('starting text must not be empty')

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    text_array = []
    context = tf.placeholder(tf.int32, [1, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    length_holder = tf.placeholder(tf.float32, ())
    temp_holder = tf.placeholder(tf.float32, ())
    top_k_holder = tf.placeholder(tf.int32, ())

    output = sample.sample_sequence(
        hparams=hparams,
        length=length_holder,
        temperature=temp_holder,
        top_k=top_k_holder,
        context=context,
        batch_size=1
    )

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)

    context_tokens = enc.encode(starting_text)
    generated = 0
    for _ in range(nsamples):
        out = sess.run(output, feed_dict={
            context: [context_tokens],
            length_holder: length,
            temp_holder: temperature,
            top_k_holder: top_k,
        })[:, len(context_tokens):]
        generated += 1
        sample_output = enc.decode(out[0])
        text_array.append(sample_output)
    return text_array


def restore_model(
        model_name='345M',
        seed=None,
        models_dir='models'):

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc: Encoder = encoder.get_encoder(model_name, models_dir)

    hparams: HParams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    np.random.seed(seed)
    tf.set_random_seed(seed)

    ph = Placeholders()
    sequence_output = sample.sample_sequence(
        hparams=hparams,
        length=ph.length,
        temperature=ph.temperature,
        top_k=ph.top_k,
        context=ph.context,
        batch_size=1
    )

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)
    return sess, hparams, sequence_output, enc, ph


class Placeholders:
    def __init__(self):
        self.context = tf.placeholder(tf.int32, [1, None])
        self.length = tf.placeholder(tf.float32, ())
        self.temperature = tf.placeholder(tf.float32, ())
        self.top_k = tf.placeholder(tf.int32, ())

    def feed_dict(self, context_tokens, length, temperature, top_k):
        return {
            self.context: [context_tokens],
            self.length: length,
            self.temperature: temperature,
            self.top_k: top_k
        }


def generate_sample(
        sess: tf.Session,
        hparams,
        sequence_output,
        enc: Encoder,
        placeholders,
        starting_text: str,
        length=None,
        temperature=1,
        top_k=0):

    if starting_text.strip() == '':
        raise ValueError('starting text must not be empty')
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    context_tokens = enc.encode(starting_text)
    feed_dict = placeholders.feed_dict(context_tokens, length, temperature, top_k)
    out = sess.run(sequence_output, feed_dict)[:, len(context_tokens):]
    sample_output = enc.decode(out[0])
    return sample_output


if __name__ == '__main__':
    fire.Fire(run_model)
