"""
Here my current code. Consume I will give you specific instructions or questions based on code. Always respond to question with only required information, no additional information needed unless asked. When a function updates, always provide complete updated code.

"""
import os
import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from typing import Dict, List, Tuple, Optional
import tensorflow as tf #type: ignore
from tensorflow.keras import layers, Model #type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay #type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy #type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction #type: ignore
from gtts import gTTS #type: ignore
from IPython.display import Audio, display
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIG = {
    'subset_ratio' : 0.001,
    'image_dir': '/home/flickr30k_images/flickr30k_images',
    'caption_file': '/home/flickr30k_images/flickr30k_images/results.csv',
    
    # GPU Utilization
    'batch_size': 128,           # Fully utilize 48GB VRAM; reduce if OOM
    'buffer_size': 1000,         # Larger shuffle buffer helps training stability
    
    # Model Capacity
    'max_length': 30,            # Reasonable for captions
    'embedding_dim': 512,        # Good for attention + LSTM
    'units': 512,                # LSTM/Attention size
    
    # Training Behavior
    'seed': 42,
    'epochs': 20,                # Slightly more for small dataset
    'patience': 4,               # Early stopping tolerance
    'learning_rate': 3e-4,       # Lower for small datasets to reduce overfitting
    'grad_clip_value': 5.0,      # Prevent exploding gradients
    
    # Vocabulary
    'vocab_min_count': 3,        # Include more words for small run
    
    # Output & Precision
    'checkpoint_dir': './checkpoints/10pct',
    'mixed_precision': False,     # RTX 6000 Ada has 4th-gen Tensor Cores—use them
}

# Set random seeds for reproducibility
tf.random.set_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# Mixed precision policy - RTX 6000 Ada has excellent mixed precision support
if CONFIG['mixed_precision']:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for RTX 6000 Ada")

# Single GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    # Enable memory growth for RTX 6000 Ada
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Use default strategy for single GPU
    strategy = tf.distribute.get_strategy()
    print(f"Using single GPU: {physical_devices[0].name}, batch size={CONFIG['batch_size']}")
else:
    print("No GPUs found, using CPU")
    strategy = tf.distribute.get_strategy()

# Constants
AUTOTUNE = tf.data.AUTOTUNE

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.captions_dict = dict()
        self.tokenizer = None
        self.vocab_size = 0
        self.train_data = []
        self.val_data = []
        self.test_data = []
    
    def load_captions(self) -> Dict[str, List[str]]:
        """Load and convert pipe-delimited Flickr-style caption file to a dict."""
        print(f"Loading captions from {self.config['caption_file']}")
        df = pd.read_csv(self.config['caption_file'], sep='|', header=None, 
                         names=['image_name', 'comment_number', 'comment'], engine='python')
        df['image_name'] = df['image_name'].str.strip()
        df['comment'] = df['comment'].str.strip()
        
        caption_map = {}
        for img, group in df.groupby('image_name'):
            caption_map[img] = group['comment'].tolist()
        
        self.captions_dict = caption_map
        print(f"Loaded {len(caption_map)} images with captions")
        return caption_map
    
    def display_samples(self, num_samples: int = 3):
        """Display random images with all their associated captions."""
        if not self.captions_dict:
            self.load_captions()

        sample_keys = random.sample(list(self.captions_dict.keys()), min(num_samples, len(self.captions_dict)))

        for key in sample_keys:
            img_path = os.path.join(self.config['image_dir'], key)
            try:
                img = Image.open(img_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(key)
                plt.show()

                for cap in self.captions_dict[key]:
                    print(f"- {cap}")
                print()
            except Exception as e:
                print(f"Error loading image {key}: {e}")
    
    def preprocess_caption(self, caption: str) -> Optional[str]:
        """Clean and format caption text."""
        if caption is None or not isinstance(caption, str):
            return None
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9.,? ]", "", caption)
        return f"<start> {caption.strip()} <end>"
    
    def prepare_captions(self, subset_ratio=1.0):
        """Process all captions and create vocabulary."""
        if not self.captions_dict:
            self.load_captions()
        
        print("Preprocessing captions...")
        all_captions = []
        for caps in self.captions_dict.values():
            for c in caps:
                p = self.preprocess_caption(c)
                if p:
                    all_captions.append(p)
        
        print(f"Total captions: {len(all_captions)}")
        word_counts = Counter(word for cap in all_captions for word in cap.split())
        valid_words = {word for word, count in word_counts.items()
                    if count >= self.config['vocab_min_count']}

        def keep_caption(caption: str) -> bool:
            words = caption.split()
            return all(w in valid_words or w in ('<start>', '<end>') for w in words)

        filtered_captions = [c for c in all_captions if keep_caption(c)]
        print(f"Filtered captions: {len(filtered_captions)}")

        # Determine max_length based on 95th percentile
        lengths = [len(c.split()) for c in filtered_captions]
        suggested_max_length = int(np.percentile(lengths, 95))
        print(f"95th percentile caption length: {suggested_max_length}")
        print(f"Max caption length in dataset: {max(lengths)}")
        
        # Update config max_length internally
        self.config['max_length'] = suggested_max_length
        print(f"Using max_length = {self.config['max_length']}")

        print("Building tokenizer...")
        tokenizer = Tokenizer(oov_token="<unk>")
        tokenizer.fit_on_texts(filtered_captions)

        for token in ['<start>', '<end>']:
            if token not in tokenizer.word_index:
                new_index = len(tokenizer.word_index) + 1
                tokenizer.word_index[token] = new_index
                tokenizer.index_word[new_index] = token

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1
        print(f"Vocabulary size: {self.vocab_size}")

        image_caption_pairs = []
        for img, caps in self.captions_dict.items():
            for c in caps:
                p = self.preprocess_caption(c)
                if p and keep_caption(p):
                    image_caption_pairs.append((img, p))

        if subset_ratio < 1.0:
            original_len = len(image_caption_pairs)
            image_caption_pairs = image_caption_pairs[:int(original_len * subset_ratio)]
            print(f"Using subset of {len(image_caption_pairs)} out of {original_len} total pairs")

        random.shuffle(image_caption_pairs)

        num_total = len(image_caption_pairs)
        train_split = int(0.8 * num_total)
        val_split = int(0.9 * num_total)
        self.train_data = image_caption_pairs[:train_split]
        self.val_data = image_caption_pairs[train_split:val_split]
        self.test_data = image_caption_pairs[val_split:]

        print(f"Dataset split: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
        return filtered_captions
        
    def encode_caption(self, caption: str) -> Tuple[np.ndarray, int]:
        """Convert caption text to sequence of token ids."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_captions first.")
        
        seq = self.tokenizer.texts_to_sequences([caption])[0]
        padded_seq = pad_sequences([seq], maxlen=self.config['max_length'], padding='post')[0]
        return padded_seq, len(seq)
    
    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def load_image(self, path: str) -> tf.Tensor:
        """Load and preprocess an image efficiently in graph mode."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, [299, 299])
        img = tf.ensure_shape(img, [299, 299, 3])
        return tf.keras.applications.inception_v3.preprocess_input(img)

    def data_generator(self, data):
        """Generator function for dataset creation."""
        for img, cap in data:
            img_path = os.path.join(self.config['image_dir'], img)
            img_tensor = self.load_image(tf.convert_to_tensor(img_path))
            token_ids, cap_len = self.encode_caption(cap)
            yield img_tensor, token_ids, cap_len
    
    def build_dataset(self, data, shuffle=True, cache=True):
        """Create a tf.data.Dataset optimized for single GPU."""
        output_signature = (
            tf.TensorSpec((299, 299, 3), tf.float32),
            tf.TensorSpec((self.config['max_length'],), tf.int32),
            tf.TensorSpec((), tf.int32)
        )

        ds = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data),
            output_signature=output_signature
        )

        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(self.config['buffer_size'])

        ds = ds.batch(self.config['batch_size'])
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def prepare_datasets(self):
        """Prepare all datasets for training/validation/testing."""
        if not self.train_data:
            self.prepare_captions()

        print("Building datasets...")
        train_ds = self.build_dataset(self.train_data)
        val_ds = self.build_dataset(self.val_data)
        test_ds = self.build_dataset(self.test_data, shuffle=False)
        
        return train_ds, val_ds, test_ds

class Encoder(Model):
    def __init__(self):
        super().__init__(name="encoder")
        # Use efficient model loading with feature extraction only
        base = tf.keras.applications.InceptionV3(
            include_top=False, 
            weights='imagenet',
            input_shape=(299, 299, 3)
        )
        base.trainable = False
        # Use specific layer for feature extraction
        output_layer = base.get_layer('mixed10').output
        self.cnn = Model(inputs=base.input, outputs=output_layer)
        self.reshape = layers.Reshape((-1, 2048))
    
    def call(self, x):
        x = self.cnn(x)
        return self.reshape(x)


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

class Decoder(Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super().__init__(name="decoder")
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)
        self.dropout = layers.Dropout(0.3)
    
    def call(self, x, features, hidden, cell):
        context, attn = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        
        hidden = tf.cast(hidden, x.dtype)
        cell = tf.cast(cell, x.dtype)

        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, state_h, state_c, attn

class ImageCaptioningModel:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.loss_fn = None
        self.ckpt_manager = None
        self.best_bleu = 0
        self.train_loss_log = []
        self.val_bleu_log = []
        self.smoothie = SmoothingFunction().method4
    
    def build_model(self):
        """Build model for single GPU - no distribution strategy needed."""
        print("Building model for single GPU...")
        self.encoder = Encoder()
        self.decoder = Decoder(
            embedding_dim=self.config['embedding_dim'], 
            units=self.config['units'], 
            vocab_size=self.processor.vocab_size
        )
        
        lr_schedule = CosineDecay(
            initial_learning_rate=self.config['learning_rate'],
            decay_steps=10000
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        # Set up checkpointing
        ckpt_dir = self.config['checkpoint_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = tf.train.Checkpoint(
            encoder=self.encoder, 
            decoder=self.decoder, 
            optimizer=self.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
        
        # Try to restore the latest checkpoint
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.ckpt_manager.latest_checkpoint}")
    
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


    @tf.function
    def train_step(self, img_tensor, target, cap_len):
        """Execute a single training step."""
        # Ensure models are built
        if self.encoder is None or self.decoder is None:
            raise ValueError("Models not built. Call build_model() first.")
        
        loss = 0.0
        batch_size = tf.shape(img_tensor)[0]
        hidden = tf.zeros((batch_size, self.config['units']))
        cell = tf.zeros_like(hidden)
        dec_input = tf.expand_dims(
            tf.repeat(self.processor.tokenizer.word_index['<start>'], batch_size), 1
        )
        
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            for t in range(1, self.config['max_length']):
                logits, hidden, cell, _ = self.decoder(dec_input, features, hidden, cell)
                loss_ = self.loss_fn(target[:, t], tf.squeeze(logits, 1))
                mask = tf.cast(target[:, t] > 0, tf.float32)
                loss += tf.reduce_sum(tf.cast(loss_, tf.float32) * mask)
                dec_input = tf.expand_dims(target[:, t], 1)
            
            # Average loss per token
            total_loss = loss / tf.reduce_sum(tf.cast(cap_len, tf.float32))
            
            # Handle mixed precision - scale loss for gradient computation
            if CONFIG['mixed_precision']:
                total_loss = tf.cast(total_loss, tf.float32)
        
        # Get trainable variables and apply gradients
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        
        # Handle mixed precision - cast gradients to float32 if needed
        if self.config['mixed_precision']:
            gradients = [tf.cast(g, tf.float32) if g is not None else None for g in gradients]
        
        # Clip gradients to prevent exploding gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config['grad_clip_value'])
        self.optimizer.apply_gradients(zip(clipped_gradients, variables))
        
        return total_loss
    
    def greedy_decode(self, image_path: str, return_attention=False):
        """Generate a caption for an image using greedy decoding."""
        img_tensor = tf.expand_dims(self.processor.load_image(image_path), 0)
        features = self.encoder(img_tensor)
        hidden = tf.zeros((1, self.config['units']))
        cell = tf.zeros_like(hidden)
        dec_input = tf.expand_dims([self.processor.tokenizer.word_index['<start>']], 0)
        
        result, alphas = [], []
        
        for _ in range(self.config['max_length']):
            logits, hidden, cell, alpha = self.decoder(dec_input, features, hidden, cell)
            predicted_id = tf.argmax(logits[0, 0]).numpy()
            
            word = self.processor.tokenizer.index_word.get(predicted_id, '')
            if word == '<end>': 
                break
                
            if word not in ('<start>', '<unk>'):
                result.append(word)
                
            alphas.append(alpha[0].numpy())
            dec_input = tf.expand_dims([predicted_id], 0)
        
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
    
    def train(self, train_ds, val_data, epochs=None):
        """Train the model with early stopping."""
        if epochs is None:
            epochs = self.config['epochs']
        
        patience = self.config['patience']
        wait = 0
        
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0.0
            step = 0
            
            # Training loop
            print(f"Epoch {epoch+1}/{epochs}")
            progbar = tf.keras.utils.Progbar(
                target=None,
                stateful_metrics=['loss']
            )
            
            for batch, (img_tensor, target, cap_len) in enumerate(train_ds):
                if batch == 0 and progbar.target is None:
                    progbar.target = len(self.processor.train_data) // self.config['batch_size'] + 1
                
                batch_loss = self.train_step(img_tensor, target, cap_len)
                total_loss += batch_loss
                progbar.update(batch + 1, values=[('loss', batch_loss)])
                step += 1
            
            # Average loss for the epoch
            avg_loss = total_loss / step
            self.train_loss_log.append(float(avg_loss))
            
            # Save checkpoint
            self.ckpt_manager.save()
            
            # Validation on a subset for speed
            print("Evaluating on validation subset...")
            validation_subset = val_data[:100]
            bleu_scores = self.evaluate_bleu(validation_subset)
            bleu4 = bleu_scores['bleu-4']
            self.val_bleu_log.append(bleu4)
            
            # Early stopping logic
            if bleu4 > self.best_bleu:
                self.best_bleu = bleu4
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, BLEU-4 = {bleu4:.4f}, Time = {time.time()-start:.2f}s", flush=True)
        
        return self.train_loss_log, self.val_bleu_log
    
    def plot_attention(self, image_path: str, caption: list, alphas: list):
        """Visualize attention weights overlaid on the source image."""
        img = np.array(Image.open(image_path).resize((224, 224)))
        fig = plt.figure(figsize=(15, 8))
        
        for t in range(len(caption)):
            ax = fig.add_subplot(3, int(np.ceil(len(caption)/3)), t+1)
            ax.set_title(caption[t])
            ax.imshow(img)
            
            alpha = np.array(alphas[t])
            attention_shape = int(np.sqrt(alpha.size))
            alpha = alpha.reshape(attention_shape, attention_shape)
            ax.imshow(alpha, cmap='viridis', alpha=0.6, extent=(0, 224, 224, 0))
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_log, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_bleu_log, label='Val BLEU-4')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-4')
        plt.title('Validation BLEU-4')
        plt.legend()
        plt.grid(True)
        
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
    
    def demo(self, image_path, filename="caption_audio.mp3"):
        """Run a full demonstration of the model."""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
            
        print(f"Generating caption for: {image_path}")
        
        # Display the image
        img = Image.open(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        # Generate caption with attention
        words, attention = self.greedy_decode(image_path, return_attention=True)
        caption = " ".join(words)
        print(f"Generated caption: {caption}")
        
        # Plot attention
        self.plot_attention(image_path, words, attention)
        
        # Generate speech
        self.speak_caption(caption, filename=filename)
        
        return caption

#--------------
processor = DataProcessor(CONFIG)
processor.load_captions()
processor.display_samples(2)
processor.prepare_captions(subset_ratio=CONFIG['subset_ratio'])
train_ds, val_ds, _ = processor.prepare_datasets()
model = ImageCaptioningModel(CONFIG, processor)
model.build_model()
model.summary()
model.train(train_ds, processor.val_data)
model.plot_history()
model.evaluate_bleu(processor.test_data[:20])
sample_img = os.path.join(CONFIG['image_dir'], processor.test_data[0][0])
model.demo(sample_img, filename='caption_audio01.mp3')
sample_pair = random.choice(processor.test_data)
sample_img = os.path.join(CONFIG['image_dir'], sample_pair[0])
model.demo(sample_img, filename='caption_audio02.mp3')
