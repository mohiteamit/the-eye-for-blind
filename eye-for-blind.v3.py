#!/usr/bin/env python
# coding: utf-8

# # Eye for Blind – Image Captioning with Attention

# # 1. Objective
# 
# Eye for Blind: An Assistive Image Captioning System with Visual Attention
# 
# This project implements a deep learning model that generates natural language descriptions of images, particularly aimed at visually impaired users. The model leverages an attention mechanism to selectively focus on image regions when generating each word, mimicking human vision.
# 
# Inspired by "Show, Attend and Tell" (Xu et al., 2015), this implementation:
# 1. Uses a CNN encoder (InceptionV3) to extract image features.
# 2. Applies additive (Bahdanau) attention during decoding.
# 3. Employs a decoder LSTM to generate captions.
# 4. Converts generated captions to speech using gTTS.

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from gtts import gTTS
from IPython.display import Audio, display
import tqdm


# In[ ]:


CONFIG = {
    # Data
    'subset_ratio': 1.0,
    'image_dir': '/home/flickr30k_images/flickr30k_images',
    'caption_file': '/home/flickr30k_images/flickr30k_images/results.csv',

    # Training
    'seed': 42,
    'epochs': 20,
    'patience': 8,
    'learning_rate': 1e-4,  # Reduced from 3e-4
    'grad_clip_value': 1.0,  # Reduced from 5.0
    'scheduled_sampling_max_prob': 0.25,
    'attention_reg_lambda': 1.0,
    'dropout_rate': 0.5,  # Increased from 0.3

    # Model Architecture
    'embedding_dim': 512,
    'units': 512,
    'max_length': 30,
    'vocab_min_count': 3,

    # Data Pipeline
    'batch_size': 128,
    'buffer_size': 10000,
    'image_size': 299,
    'resize_size': 342,

    # Inference
    'beam_size': 5,
    'length_penalty': 0.7,
    'focus_threshold': 0.5,  # For attention diagnostics

    # System
    'mixed_precision': False,
    'enable_checkpointing': True,  # New flag
    'checkpoint_dir': './checkpoints/',
    'restore_checkpoint': False,  # New flag
}


# In[ ]:


# Set random seeds
tf.random.set_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# Mixed precision
if CONFIG['mixed_precision']:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.get_strategy()
else:
    strategy = tf.distribute.get_strategy()

AUTOTUNE = tf.data.AUTOTUNE


# In[ ]:


class DataProcessor:
    def __init__(self, config):
        self.config = config.copy()  # Ensure local copy
        self.captions_dict = dict()
        self.tokenizer = None
        self.vocab_size = 0
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def load_captions(self):
        df = pd.read_csv(self.config['caption_file'], sep='|', header=None, 
                         names=['image_name', 'comment_number', 'comment'], engine='python')
        df['image_name'] = df['image_name'].str.strip()
        df['comment'] = df['comment'].str.strip()
        self.captions_dict = {img: group['comment'].tolist() for img, group in df.groupby('image_name')}
        return self.captions_dict

    def preprocess_caption(self, caption):
        if not isinstance(caption, str):
            return None
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9.,? ]", "", caption)
        return f"<start> {caption.strip()} <end>"

    def prepare_captions(self):
        all_captions = []
        for caps in self.captions_dict.values():
            for c in caps:
                p = self.preprocess_caption(c)
                if p: all_captions.append(p)

        word_counts = Counter(w for cap in all_captions for w in cap.split())
        valid_words = {w for w, cnt in word_counts.items() if cnt >= self.config['vocab_min_count']}
        filtered = [c for c in all_captions if all(w in valid_words or w in ('<start>', '<end>') for w in c.split())]

        tokenizer = Tokenizer(oov_token="<unk>", filters='', lower=True)
        tokenizer.fit_on_texts(filtered)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1

        pairs = []
        for img, caps in self.captions_dict.items():
            for c in caps:
                p = self.preprocess_caption(c)
                if p and all(w in valid_words or w in ('<start>', '<end>') for w in p.split()):
                    pairs.append((img, p))

        if self.config['subset_ratio'] < 1.0:
            pairs = pairs[:int(len(pairs) * self.config['subset_ratio'])]

        random.shuffle(pairs)
        n = len(pairs)
        self.train_data, self.val_data, self.test_data = (
            pairs[:int(0.8*n)], pairs[int(0.8*n):int(0.9*n)], pairs[int(0.9*n):])
        return filtered

    def encode_caption(self, caption):
        seq = self.tokenizer.texts_to_sequences([caption])[0]
        padded_seq = pad_sequences([seq], maxlen=self.config['max_length'], padding='post')[0]
        return padded_seq, len(seq)

    @tf.function
    def _base_decode(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    @tf.function
    def load_image_train(self, path):
        img = self._base_decode(path)
        img = tf.image.random_flip_left_right(img)
        shape = tf.shape(img)[:2]
        scale = self.config['resize_size'] / tf.cast(tf.reduce_min(shape), tf.float32)
        new_hw = tf.cast(tf.cast(shape, tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, new_hw)
        img = tf.image.random_crop(img, size=[self.config['image_size'], self.config['image_size'], 3])
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return tf.ensure_shape(img, [self.config['image_size'], self.config['image_size'], 3])

    @tf.function
    def load_image_eval(self, path):
        img = self._base_decode(path)
        shape = tf.shape(img)[:2]
        scale = self.config['resize_size'] / tf.cast(tf.reduce_min(shape), tf.float32)
        new_hw = tf.cast(tf.cast(shape, tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, new_hw)
        img = tf.image.resize_with_crop_or_pad(img, self.config['image_size'], self.config['image_size'])
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return tf.ensure_shape(img, [self.config['image_size'], self.config['image_size'], 3])

    def build_dataset(self, data, shuffle=True):
        def generator():
            for img, cap in data:
                img_path = os.path.join(self.config['image_dir'], img)
                img_tensor = self.load_image_train(tf.convert_to_tensor(img_path))
                token_ids, cap_len = self.encode_caption(cap)
                yield img_tensor, token_ids, cap_len

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec((self.config['image_size'], self.config['image_size'], 3), tf.float32),
                tf.TensorSpec((self.config['max_length'],), tf.int32),
                tf.TensorSpec((), tf.int32)
            )
        )

        if shuffle:
            ds = ds.shuffle(self.config['buffer_size'])
        ds = ds.batch(self.config['batch_size'])
        return ds.prefetch(AUTOTUNE)


# In[ ]:


class Encoder(Model):
    def __init__(self, config):
        super().__init__(name="encoder")
        self.config = config.copy()
        base = tf.keras.applications.InceptionV3(
            include_top=False, weights='imagenet',
            input_shape=(self.config['image_size'], self.config['image_size'], 3))
        base.trainable = False
        self.cnn = Model(inputs=base.input, outputs=base.get_layer('mixed10').output)
        self.reshape = layers.Reshape((-1, 2048))

    def unfreeze_top_layers(self, n=2):
        for layer in self.cnn.layers[-n:]:
            layer.trainable = True

    def call(self, x):
        x = self.cnn(x)
        return self.reshape(x)


# In[ ]:


class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__(name="attention")
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_time)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1)


# In[ ]:


class Decoder(Model):
    def __init__(self, config, vocab_size):
        super().__init__(name="decoder")
        self.config = config.copy()
        self.units = self.config['units']

        self.embedding = layers.Embedding(vocab_size, self.config['embedding_dim'])
        self.attention = BahdanauAttention(self.units)
        self.f_beta = layers.Dense(1, activation="sigmoid")
        self.lstm = layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.dropout = layers.Dropout(self.config['dropout_rate'])
        self.deep_proj = layers.Dense(self.units * 2)
        self.fc = layers.Dense(vocab_size)

    def call(self, x, features, hidden, cell):
        context, alpha = self.attention(features, hidden)
        context = self.f_beta(hidden) * context

        x = self.embedding(x)
        lstm_input = tf.concat([tf.expand_dims(context, 1), x], -1)

        hidden = tf.cast(hidden, lstm_input.dtype)
        cell = tf.cast(cell, lstm_input.dtype)

        lstm_out, h_t, c_t = self.lstm(lstm_input, initial_state=[hidden, cell])
        lstm_out = tf.squeeze(lstm_out, 1)

        proj = self.deep_proj(tf.concat([lstm_out, context], -1))
        proj = tf.reshape(proj, (-1, self.units, 2))
        maxout = tf.reduce_max(proj, axis=-1)
        maxout = self.dropout(maxout)

        logits = self.fc(maxout)
        return tf.expand_dims(logits, 1), h_t, c_t, alpha


# In[ ]:


class ImageCaptioningModel:
    def __init__(self, config, processor):
        self.config = config.copy()
        self.processor = processor
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.loss_fn = None
        self.ckpt_manager = None
        self.best_bleu = 0.0
        self.train_loss_log = []
        self.train_bleu_log = []
        self.val_bleu_log = []
        self.bleu_subset_idx = None
        self.smoothie = SmoothingFunction().method4
        self.ss_prob = 0.0

    def build_model(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, self.processor.vocab_size)

        lr_schedule = CosineDecay(
            initial_learning_rate=self.config['learning_rate'],
            decay_steps=10000
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        if self.config['enable_checkpointing']:
            ckpt_dir = self.config['checkpoint_dir']
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = tf.train.Checkpoint(
                encoder=self.encoder,
                decoder=self.decoder,
                optimizer=self.optimizer
            )
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

            if self.config['restore_checkpoint'] and self.ckpt_manager.latest_checkpoint:
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print(f"Restored from {self.ckpt_manager.latest_checkpoint}")

    def train_step(self, img_tensor, target, cap_len):
        batch_size = tf.shape(img_tensor)[0]
        hidden = tf.zeros((batch_size, self.config['units']))
        cell = tf.zeros_like(hidden)
        start_tok = self.processor.tokenizer.word_index['<start>']
        dec_input = tf.expand_dims(tf.repeat(start_tok, batch_size), 1)

        attention_accum = None
        total_ce_loss = 0.0

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for t in tf.range(1, self.config['max_length']):
                logits, hidden, cell, alpha = self.decoder(dec_input, features, hidden, cell)
                attention_accum = alpha if attention_accum is None else attention_accum + alpha

                ce_t = self.loss_fn(target[:, t], tf.squeeze(logits, 1))
                mask = tf.cast(target[:, t] > 0, tf.float32)
                total_ce_loss += tf.reduce_sum(ce_t * mask)

                pred_ids = tf.argmax(logits, -1, output_type=tf.int32)
                pred_ids = tf.squeeze(pred_ids, -1)
                ss_mask = tf.random.uniform((batch_size,)) < self.ss_prob
                next_ids = tf.where(ss_mask, pred_ids, target[:, t])
                dec_input = tf.expand_dims(next_ids, 1)

            total_tokens = tf.reduce_sum(tf.cast(cap_len, tf.float32))
            ce_loss = total_ce_loss / total_tokens
            reg_loss = tf.reduce_mean(tf.square(1.0 - attention_accum))
            loss = ce_loss + self.config['attention_reg_lambda'] * reg_loss

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config['grad_clip_value'])
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def plot_attention(self, image_path, caption, alphas):
        img = np.array(Image.open(image_path))
        orig_h, orig_w = img.shape[:2]
        fig = plt.figure(figsize=(15, 8))

        # Attention grid is 8x8 for InceptionV3 mixed10
        grid_size = int(np.sqrt(alphas[0].shape[0]))

        for t in range(len(caption)):
            ax = fig.add_subplot(3, int(np.ceil(len(caption)/3)), t+1)
            ax.set_title(caption[t])
            ax.imshow(img)

            alpha = alphas[t].reshape(grid_size, grid_size)
            alpha_resized = np.array(Image.fromarray(alpha).resize(
                (orig_w, orig_h), Image.BILINEAR))
            ax.imshow(alpha_resized, cmap='viridis', alpha=0.6, extent=(0, orig_w, orig_h, 0))
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def analyze_generations(self, data, max_samples=100):
        word_counts = Counter()
        unk_count = 0
        data_subset = random.sample(data, min(max_samples, len(data)))

        for img, _ in tqdm.tqdm(data_subset):
            caption = self.greedy_decode(os.path.join(self.config['image_dir'], img))
            word_counts.update(caption)
            unk_count += sum(1 for w in caption if w == '<unk>')

        print("\nTop 20 generated words:")
        for word, count in word_counts.most_common(20):
            print(f"{word}: {count}")

        print(f"\nUNK tokens: {unk_count} ({unk_count/sum(word_counts.values()):.1%})")
        return word_counts

    def train(self, train_ds, val_data, epochs=None):
        if epochs is None:
            epochs = self.config['epochs']

        if self.bleu_subset_idx is None:
            total_train = len(self.processor.train_data)
            subset_size = min(200, total_train)
            self.bleu_subset_idx = random.sample(range(total_train), subset_size)

        def _subset(data, idx):
            return [data[i] for i in idx]

        patience = self.config['patience']
        wait = 0
        self.ss_max_prob = self.config['scheduled_sampling_max_prob']

        for epoch in range(epochs):
            self.ss_prob = self.ss_max_prob * epoch / max(1, epochs - 1)
            print(f"\nEpoch {epoch+1}/{epochs} (ε={self.ss_prob:.3f})")

            start = time.time()
            total_loss, step = 0.0, 0
            progbar = tf.keras.utils.Progbar(None)

            for batch, (img_tensor, target, cap_len) in enumerate(train_ds):
                batch_loss = self.train_step(img_tensor, target, cap_len)
                total_loss += batch_loss
                progbar.update(batch + 1, [('loss', batch_loss)])
                step += 1

            avg_loss = total_loss / step
            self.train_loss_log.append(float(avg_loss))

            train_subset = _subset(self.processor.train_data, self.bleu_subset_idx)
            train_bleu = self.evaluate_bleu(train_subset)['bleu-4']
            self.train_bleu_log.append(train_bleu)

            val_bleu = self.evaluate_bleu(val_data)['bleu-4']
            self.val_bleu_log.append(val_bleu)

            if self.config['enable_checkpointing']:
                self.ckpt_manager.save()

            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} "
                  f"train-BLEU={train_bleu:.4f} val-BLEU={val_bleu:.4f} "
                  f"time={time.time()-start:.1f}s", flush=True)

    def summary(self):
        """Print model summaries for Encoder, Attention, and Decoder."""
        print("Building model summaries...")

        # Dummy inputs
        dummy_image = tf.random.uniform((1, 299, 299, 3))
        dummy_features = tf.random.uniform((1, 64, 2048))
        dummy_hidden = tf.zeros((1, self.config['units']))
        dummy_cell = tf.zeros((1, self.config['units']))
        dummy_token = tf.zeros((1, 1), dtype=tf.int32)

        # --- Encoder Summary ---
        print("\nEncoder Summary:")
        self.encoder(dummy_image)
        self.encoder.summary()

        # --- Bahdanau Attention Summary ---
        print("\nBahdanau Attention Summary:")
        attention_layer = BahdanauAttention(self.config['units'])
        features_input = tf.keras.Input(shape=(64, 2048), name="features")
        hidden_input = tf.keras.Input(shape=(self.config['units'],), name="hidden")
        context_vector, attn_weights = attention_layer(features_input, hidden_input)
        attention_model = tf.keras.Model(inputs=[features_input, hidden_input], outputs=[context_vector, attn_weights])
        attention_model.summary()

        # --- Decoder Summary ---
        print("\nDecoder Summary:")
        self.decoder(dummy_token, dummy_features, dummy_hidden, dummy_cell)
        self.decoder.summary()

    def beam_search_decode(self,
                           image_path: str,
                           beam_size: int = 5,
                           length_penalty: float = 0.7,
                           return_attention: bool = False):
        """Beam-search with deterministic crop."""
        img_tensor = tf.expand_dims(
            self.processor.load_image_eval(tf.convert_to_tensor(image_path)), 0
        )
        base_features = self.encoder(img_tensor)       # (1,L,2048)

        start_id = self.processor.tokenizer.word_index['<start>']
        end_id   = self.processor.tokenizer.word_index['<end>']

        beams = [{'seq':[start_id],
                  'score':0.0,
                  'hidden':tf.zeros((1,self.config['units'])),
                  'cell':tf.zeros((1,self.config['units'])),
                  'alphas':[]}]

        completed = []
        for _ in range(self.config['max_length']):
            candidates = []
            for b in beams:
                last_id = b['seq'][-1]
                if last_id == end_id:
                    completed.append(b); continue
                dec_in = tf.expand_dims([last_id], 0)
                logits, h, c, alpha = self.decoder(dec_in, base_features,
                                                   b['hidden'], b['cell'])
                log_probs = tf.nn.log_softmax(logits[0,0])
                top_ids = tf.math.top_k(log_probs, k=beam_size).indices.numpy()
                for tok in top_ids:
                    tok = int(tok)
                    candidates.append({
                        'seq':   b['seq']+[tok],
                        'score': b['score']+float(log_probs[tok]),
                        'hidden':h,
                        'cell':  c,
                        'alphas':b['alphas']+[alpha[0].numpy()]})
            if not candidates: break
            def lp(b): return b['score']/(len(b['seq'])**length_penalty)
            candidates.sort(key=lp, reverse=True)
            beams = candidates[:beam_size]
            if len(completed) >= beam_size: break

        best = max(completed+beams,
                   key=lambda b: b['score']/(len(b['seq'])**length_penalty))
        words = [self.processor.tokenizer.index_word.get(i,'')
                 for i in best['seq']
                 if self.processor.tokenizer.index_word.get(i,'') not in
                 ('<start>','<end>','<unk>')]
        return (words, best['alphas']) if return_attention else words

    def greedy_decode(self, image_path: str, return_attention=False):
        """Generate caption via greedy decoding (deterministic crop)."""
        img_tensor = tf.expand_dims(
            self.processor.load_image_eval(tf.convert_to_tensor(image_path)), 0
        )

        features = self.encoder(img_tensor)
        hidden = tf.zeros((1, self.config['units']))
        cell   = tf.zeros_like(hidden)
        dec_input = tf.expand_dims(
            [self.processor.tokenizer.word_index['<start>']], 0
        )

        result, alphas = [], []
        for _ in range(self.config['max_length']):
            logits, hidden, cell, alpha = self.decoder(
                dec_input, features, hidden, cell
            )
            pred_id = tf.argmax(logits[0, 0]).numpy()
            word = self.processor.tokenizer.index_word.get(pred_id, '')
            if word == '<end>':
                break
            if word not in ('<start>', '<unk>'):
                result.append(word)
            alphas.append(alpha[0].numpy())
            dec_input = tf.expand_dims([pred_id], 0)

        return (result, alphas) if return_attention else result

    def evaluate_bleu(self, test_data, max_samples=None):
        """Calculate BLEU scores on test data."""
        refs, hyps = [], []
        data_to_eval = test_data[:max_samples] if max_samples else test_data

        for img_name, _ in tqdm.tqdm(data_to_eval):
            image_path = os.path.join(self.config['image_dir'], img_name)
            hyp = self.greedy_decode(image_path)

            # Process ground truth captions
            gt = [self.processor.preprocess_caption(c).split() for c in self.processor.captions_dict[img_name][:5]]
            gt = [[w for w in cap if w not in ('<start>', '<end>')] for cap in gt]

            refs.append(gt)
            hyps.append(hyp)

        # Calculate BLEU scores for different n-grams
        bleu_scores = {}
        for i in range(1, 5):
            weights = tuple([1.0/i]*i + [0.0]*(4-i))
            score = corpus_bleu(refs, hyps, weights=weights, smoothing_function=self.smoothie)
            bleu_scores[f'bleu-{i}'] = score
            print(f"BLEU-{i}: {score:.4f}")

        return bleu_scores

    def plot_history(self):
        """Plot loss curve **and** both train/val BLEU-4 curves."""
        plt.figure(figsize=(14, 5))

        # --- left: training loss ---
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_log, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.legend()

        # --- right: BLEU-4 ---
        plt.subplot(1, 2, 2)
        if self.train_bleu_log:
            plt.plot(self.train_bleu_log, label='Train BLEU-4')
        plt.plot(self.val_bleu_log,   label='Val BLEU-4')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-4')
        plt.title('BLEU-4 Scores')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def speak_caption(self, caption: str, filename="caption_audio.mp3"):
        """Generate speech audio from caption text."""
        if not caption:
            print("Empty caption, nothing to speak")
            return

        tts = gTTS(text=caption, lang='en')
        tts.save(filename)
        display(Audio(filename))
        print(f"Audio saved to {filename}")

    def demo(self,
             image_path: str,
             filename: str = "caption_audio.mp3",
             beam_size: int = 5,
             length_penalty: float = 0.7):
        """
        End-to-end demo (beam-search inference) in the following order:
          1. Original image
          2. Ground-truth captions
          3. Generated caption
          4. Audio playback
          5. Attention heat-maps
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return

        # ---------- 1. original image ----------
        img = Image.open(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # ---------- 2. ground-truth captions ----------
        img_name = os.path.basename(image_path)
        gt_caps = self.processor.captions_dict.get(img_name, [])
        if gt_caps:
            print("Ground-truth captions:")
            for cap in gt_caps:
                print(f"- {cap}")
        else:
            print("No ground-truth captions found.")

        # ---------- 3. caption generation ----------
        words, attention = self.beam_search_decode(
            image_path,
            beam_size=beam_size,
            length_penalty=length_penalty,
            return_attention=True
        )
        caption = " ".join(words)
        print("\nGenerated caption:")
        print(caption)

        # ---------- 4. audio ----------
        self.speak_caption(caption, filename=filename)

        # ---------- 5. attention plot ----------
        self.plot_attention(image_path, words, attention)

    def prime_dataset(self, ds, steps: int = None) -> None:
        """
        Pre-fill a tf.data shuffle buffer so the first training epoch
        starts without the usual “Filling up shuffle buffer …” pause.

        Args
        ----
        ds    : the *un-iterated* tf.data.Dataset you’ll pass to train()
        steps : number of iterator steps to advance; default uses
                buffer_size // batch_size + 1 from config.
        """
        if steps is None:
            steps = self.config['buffer_size'] // self.config['batch_size'] + 1

        it = iter(ds)
        for _ in range(steps):
            try:
                next(it)
            except StopIteration:  # dataset shorter than requested priming
                break

    def fine_tune_cnn(self,
                      train_ds,
                      val_data,
                      layers_to_unfreeze: int = 2,
                      lr: float = 1e-5,
                      epochs: int = 1):
        """
        Phase-2 fine-tuning of the top Inception blocks.
        Call after initial caption training for an extra accuracy bump.
        """
        print(f"\nUnfreezing top {layers_to_unfreeze} Inception blocks …")
        self.encoder.unfreeze_top_layers(layers_to_unfreeze)

        # New, low learning-rate optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"Fine-tuning CNN for {epochs} epoch(s) at lr={lr} …")
        self.train(train_ds, val_data, epochs=epochs)

        print("CNN fine-tune finished.")


# In[ ]:


# Initialize components
processor = DataProcessor(CONFIG)
processor.load_captions()
processor.prepare_captions()

# Prepare datasets
train_ds = processor.build_dataset(processor.train_data)
val_ds = processor.build_dataset(processor.val_data, shuffle=False)

# Build and train model
model = ImageCaptioningModel(CONFIG, processor)
model.build_model()

# Optional: Analyze before training
model.analyze_generations(processor.train_data)

# Train the model
model.train(train_ds, processor.val_data)

# Optional: Fine-tune CNN
model.fine_tune_cnn(train_ds, processor.val_data, 
                   layers_to_unfreeze=8, lr=1e-5, epochs=5)

# Evaluate
model.evaluate_bleu(processor.test_data)

