# 2. Data understanding
IMAGE_DIR = 'data/flickr30k_images'                            # directory containing images
CAPTION_FILE = 'data/flickr30k_images/results.csv'             # pipe-separated file

import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Enable memory growth for GPUs
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from PIL import Image
import pandas as pd

def load_captions(path: str) -> dict:
    """Load and convert pipe-delimited Flickr-style caption file to a dict."""
    df = pd.read_csv(path, sep='\|', header=None, names=['image_name', 'comment_number', 'comment'], engine='python')
    df['image_name'] = df['image_name'].str.strip()
    df['comment'] = df['comment'].str.strip()
    caption_map = {}
    for img, group in df.groupby('image_name'):
        caption_map[img] = group['comment'].tolist()
    return caption_map

captions_dict = load_captions(CAPTION_FILE)

def display_samples(image_dir: str, captions: dict, num_samples: int = 3):
    sample_keys = random.sample(list(captions.keys()), num_samples)
    for key in sample_keys:
        img_path = os.path.join(image_dir, key)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(key)
        plt.show()
        for cap in captions[key][:5]:
            print(f"- {cap}")
        print()

display_samples(IMAGE_DIR, captions_dict)

# 3. Data Pre-processing
import re
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

AUTOTUNE = tf.data.AUTOTUNE
MAX_LEN = 30
BATCH_SIZE = 64
SEED = 42

def preprocess_caption(caption: str) -> str:
    if caption is None or not isinstance(caption, str):
        return None
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9 ]", "", caption)
    return f"<start> {caption.strip()} <end>"

# Prepare captions and vocabulary
all_captions = []
for caps in captions_dict.values():
    for c in caps:
        p = preprocess_caption(c)
        if p:
            all_captions.append(p)

word_counts = Counter(word for cap in all_captions for word in cap.split())

def keep_caption(caption: str) -> bool:
    words = caption.split()
    for w in words:
        if w in ('<start>', '<end>'):
            continue
        if word_counts.get(w, 0) < 5:
            return False
    return True

filtered_captions = [c for c in all_captions if keep_caption(c)]

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(filtered_captions)
# Ensure special tokens
for token in ['<start>', '<end>']:
    if token not in tokenizer.word_index:
        tokenizer.word_index[token] = len(tokenizer.word_index) + 1

vocab_size = len(tokenizer.word_index) + 1

def encode_caption(caption: str):
    seq = tokenizer.texts_to_sequences([caption])[0]
    return pad_sequences([seq], maxlen=MAX_LEN, padding='post')[0], len(seq)

def load_image(path: str) -> tf.Tensor:
    """Load and preprocess an image with aspect-preserving resize and padding."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_with_pad(img, 299, 299)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# Create train/val/test splits
image_caption_pairs = []
for img, caps in captions_dict.items():
    for c in caps:
        p = preprocess_caption(c)
        if p and keep_caption(p):
            image_caption_pairs.append((img, p))

random.seed(SEED)
random.shuffle(image_caption_pairs)
num_total = len(image_caption_pairs)
train_split = int(0.8 * num_total)
val_split = int(0.9 * num_total)
train_data = image_caption_pairs[:train_split]
val_data = image_caption_pairs[train_split:val_split]
test_data = image_caption_pairs[val_split:]

def data_generator(data):
    for img, cap in data:
        img_tensor = load_image(os.path.join(IMAGE_DIR, img))
        token_ids, cap_len = encode_caption(cap)
        yield img_tensor, token_ids, cap_len

output_signature = (
    tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

def build_tf_dataset(data):
    return (tf.data.Dataset
            .from_generator(lambda: data_generator(data), output_signature=output_signature)
            .shuffle(1024)
            .padded_batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

# train_ds = build_tf_dataset(train_data)
# val_ds = build_tf_dataset(val_data)

train_ds = build_tf_dataset(train_data[:256])
val_ds = build_tf_dataset(val_data[:64])
test_data = test_data[:10]

# 4. Model Building
from tensorflow.keras import layers, Model

class Encoder(Model):
    def __init__(self):
        super().__init__()
        base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        base.trainable = False
        self.cnn = Model(inputs=base.input, outputs=base.get_layer('mixed10').output)
        self.reshape = layers.Reshape((-1, 2048))

    def call(self, x):
        x = self.cnn(x)
        return self.reshape(x)

class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
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
        super().__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden, cell):
        context, attn = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        logits = self.fc(output)
        return logits, state_h, state_c, attn

# 5. Training
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gtts import gTTS
from IPython.display import Audio, display

smoothie = SmoothingFunction().method4

def decode_tokens(tokens):
    words = [tokenizer.index_word.get(idx, '') for idx in tokens if idx != 0]
    return ' '.join([w for w in words if w not in ('<start>', '<end>')])

def greedy_decode(image_path: str, return_attention=False):
    img_tensor = tf.expand_dims(load_image(image_path), 0)
    features = encoder(img_tensor)
    hidden = tf.zeros((1, 512))
    cell = tf.zeros_like(hidden)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    result, alphas = [], []
    for _ in range(MAX_LEN):
        logits, hidden, cell, alpha = decoder(dec_input, features, hidden, cell)
        predicted_id = tf.argmax(logits[0,0]).numpy()
        word = tokenizer.index_word.get(predicted_id, '')
        if word == '<end>': break
        result.append(word)
        alphas.append(alpha[0].numpy())
        dec_input = tf.expand_dims([predicted_id], 0)

    return (result, alphas) if return_attention else result

def evaluate_bleu(test_data):
    refs, hyps = [], []
    for img_name, caps in tqdm.tqdm(test_data):
        image_path = os.path.join(IMAGE_DIR, img_name)
        hyp = greedy_decode(image_path)
        gt = [preprocess_caption(c).split() for c in captions_dict[img_name][:5]]
        gt = [[w for w in cap if w not in ('<start>', '<end>')] for cap in gt]
        refs.append(gt)
        hyps.append(hyp)
    for i in range(1,5):
        weights = tuple([1.0/i]*i + [0.0]*(4-i))
        score = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothie)
        print(f"BLEU-{i}: {score:.4f}")

def plot_attention(image_path: str, caption: list, alphas: list):
    img = np.array(Image.open(image_path).resize((224,224)))
    fig = plt.figure(figsize=(15,8))
    for t in range(len(caption)):
        ax = fig.add_subplot(3, int(np.ceil(len(caption)/3)), t+1)
        ax.set_title(caption[t])
        ax.imshow(img)
        alpha = np.array(alphas[t]).reshape(8,8)
        ax.imshow(alpha, cmap='viridis', alpha=0.6, extent=(0,224,224,0))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def speak_caption(caption: str, filename="tts_output.mp3"):
    tts = gTTS(text=caption, lang='en')
    tts.save(filename)
    display(Audio(filename))

import time
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy

encoder = Encoder()
decoder = Decoder(embedding_dim=512, units=512, vocab_size=vocab_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=CosineDecay(1e-3, decay_steps=10000))
loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

ckpt_dir = './checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

train_loss_log = []
val_bleu_log = []

@tf.function
def train_step(img_tensor, target, cap_len):
    loss = 0.0
    hidden = tf.zeros((img_tensor.shape[0], 512))
    cell = tf.zeros_like(hidden)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * img_tensor.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for t in range(1, MAX_LEN):
            logits, hidden, cell, _ = decoder(dec_input, features, hidden, cell)
            loss_ = loss_fn(target[:, t], tf.squeeze(logits, 1))
            mask = tf.cast(target[:, t] > 0, tf.float32)
            loss += tf.reduce_sum(loss_ * mask)
            dec_input = tf.expand_dims(target[:, t], 1)

    variables = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, variables))
    return loss / tf.reduce_sum(tf.cast(cap_len, tf.float32))

EPOCHS = 10
patience = 3
best_bleu = 0
wait = 0

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0.0
    for img_tensor, target, cap_len in train_ds:
        total_loss += train_step(img_tensor, target, cap_len)
    train_loss_log.append(float(total_loss))
    ckpt_mgr.save()

    # Validation
    refs, hyps = [], []
    for img_name, caps in val_data[:100]:
        image_path = os.path.join(IMAGE_DIR, img_name)
        hyp = greedy_decode(image_path)
        gt = [preprocess_caption(c).split() for c in captions_dict[img_name][:5]]
        gt = [[w for w in cap if w not in ('<start>', '<end>')] for cap in gt]
        refs.append(gt)
        hyps.append(hyp)
    bleu4 = corpus_bleu(refs, hyps, smoothing_function=smoothie)
    val_bleu_log.append(bleu4)

    if bleu4 > best_bleu:
        best_bleu = bleu4
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, BLEU-4: {bleu4:.4f}, Time: {time.time()-start:.2f}s", flush=True)

print("Final evaluation on full test set:")
evaluate_bleu(test_data)

# Plot loss and BLEU
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_log, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(val_bleu_log, label='Val BLEU-4')
plt.xlabel('Epoch')
plt.ylabel('BLEU-4')
plt.title('Validation BLEU-4')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Inference & Evaluation

sample_img = os.path.join(IMAGE_DIR, test_data[0][0])
result_words, attention = greedy_decode(sample_img, return_attention=True)
print("Generated caption:", " ".join(result_words))

caption_text = " ".join(result_words)  # assuming result_words from greedy_decode

# Convert to speech and save
tts = gTTS(text=caption_text, lang='en')
tts.save("caption_audio.mp3")

# Display inline audio player in Jupyter
display(Audio("caption_audio.mp3"))
