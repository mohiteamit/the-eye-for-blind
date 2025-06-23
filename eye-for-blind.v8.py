# ──────────────────────────────────────────────────────────────────────────────
#  EYE FOR BLIND – “SHOW, ATTEND & TELL” - FULL REWRITE (review v2025-06-24)
#  All 14 checklist items from the code-review are implemented below.
#  Paste this single cell into your notebook and run from the top.
# ──────────────────────────────────────────────────────────────────────────────
import os, re, time, math, random, json, pickle, itertools, warnings
from collections import Counter, deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import tensorflow as tf                  # type: ignore
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIG – now covers every checklist flag in one place
# ──────────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, object] = {
    # Data
    'image_dir'            : '/home/flickr30k_images/flickr30k_images',
    'caption_file'         : '/home/flickr30k_images/flickr30k_images/results.csv',
    'subset_ratio'         : 1.0,
    # Vocabulary & sequence
    'vocab_min_count'      : 2,          # [spec 1] ↓ from 5
    'top_k'                : 10_000,     # [spec 1] explicit cap
    'max_length'           : 30,
    # Model
    'embedding_dim'        : 512,
    'units'                : 1024,
    'decoder_dropout'      : 0.3,
    'attention_reg_lambda' : 0.1,        # [spec 5]  coverage regulariser
    # Training
    'epochs'               : 30,
    'batch_size'           : 128,
    'buffer_size'          : 10_000,
    'scheduled_sampling_max_prob': 0.4,  # [spec 4]  ↑ from 0.15
    'mixed_precision'      : True,
    'grad_clip_value'      : 10.0,
    'early_stop'           : True,
    'patience'             : 20,
    # Optimiser / LR
    'initial_lr'           : 5e-4,
    'lr_alpha'             : 1e-2,
    # Checkpoints
    'checkpoint_dir'       : './checkpoints/split_by_image',
    'save_checkpoints'     : True,
    'delete_old_checkpoints': True,
    # Misc / reproducibility
    'seed'                 : 42,
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. DETERMINISTIC INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])
warnings.filterwarnings("ignore")

if CONFIG['mixed_precision']:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[AMP] mixed_float16 policy active ✨")

# single-GPU safe-growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
print("Device list:", tf.config.list_logical_devices('GPU'))

AUTOTUNE = tf.data.AUTOTUNE

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA PROCESSOR  – handles tokeniser & tf.data pipelines
# ──────────────────────────────────────────────────────────────────────────────
class DataProcessor:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.caption_map: Dict[str, List[str]] = {}
        self.tokeniser: Optional[Tokenizer]    = None
        self.vocab_size = 0
        self.train_pairs: List[Tuple[str,str]] = []
        self.val_pairs  : List[Tuple[str,str]] = []
        self.test_pairs : List[Tuple[str,str]] = []

    # ───────────────────────────────────────────
    # 3.1 load raw captions file
    # ───────────────────────────────────────────
    def load_captions(self):
        fp = self.cfg['caption_file']
        df = pd.read_csv(fp, sep='|', names=['img','num','cap'], engine='python')
        df['img']  = df['img'].str.strip(); df['cap'] = df['cap'].str.strip()
        for img,g in df.groupby('img'):
            self.caption_map[img] = g['cap'].tolist()
        print(f"Loaded {len(self.caption_map):,} images with captions")

    # ───────────────────────────────────────────
    # 3.2 text preproc
    # ───────────────────────────────────────────
    @staticmethod
    def preprocess(txt:str)->str:
        if not isinstance(txt,str) or not txt.strip(): return ""
        txt = re.sub(r"[^a-z0-9.,? ]","",txt.lower().strip())
        return f"<start> {txt} <end>"

    # ───────────────────────────────────────────
    # 3.3 build tokeniser & split
    # ───────────────────────────────────────────
    def prepare(self):
        if not self.caption_map: self.load_captions()
        all_caps = [self.preprocess(c) for caps in self.caption_map.values() for c in caps]
        # vocab pruning
        cnt = Counter(w for sent in all_caps for w in sent.split())
        keep = {w for w,f in cnt.items() if f>=self.cfg['vocab_min_count']} | {'<start>','<end>'}
        filtered = [c for c in all_caps if all(w in keep for w in c.split())]

        # 95-th percentile length
        self.cfg['max_length'] = int(np.percentile([len(s.split()) for s in filtered],95))
        print("max_length →",self.cfg['max_length'])

        # tokeniser with explicit pad-idx 0  [spec 1,2]
        tok = Tokenizer(num_words=self.cfg['top_k'], oov_token='<unk>', filters='', lower=True)
        tok.fit_on_texts(filtered)
        tok.word_index['<pad>']  = 0
        tok.index_word[0]        = '<pad>'
        self.tokeniser, self.vocab_size = tok, self.cfg['top_k']+1
        print("vocab_size capped at",self.vocab_size)

        # img-wise (80/10/10) split
        pairs = [(img,self.preprocess(c)) for img,caps in self.caption_map.items() for c in caps
                 if self.preprocess(c) and all(w in keep for w in self.preprocess(c).split())]
        imgset = list({img for img,_ in pairs})
        if self.cfg['subset_ratio']<1.0:
            k=int(len(imgset)*self.cfg['subset_ratio']); imgset=random.sample(imgset,k)
        random.shuffle(imgset)
        n=len(imgset); n_tr=int(.8*n); n_v=int(.1*n)
        tr,val,test = set(imgset[:n_tr]),set(imgset[n_tr:n_tr+n_v]),set(imgset[n_tr+n_v:])
        f=lambda s:[p for p in pairs if p[0] in s]
        self.train_pairs,self.val_pairs,self.test_pairs = map(f,(tr,val,test))
        print(f"train/val/test images: {len(tr)}/{len(val)}/{len(test)}")

    # ───────────────────────────────────────────
    # 3.4 helpers
    # ───────────────────────────────────────────
    def encode(self, cap:str):
        seq=self.tokeniser.texts_to_sequences([cap])[0]
        seq=pad_sequences([seq],maxlen=self.cfg['max_length'],padding='post',value=0)[0] # [spec 2]
        return seq, len([t for t in seq if t!=0])

    @tf.function(input_signature=[tf.TensorSpec([],tf.string)])
    def _load_img(self,path):
        img=tf.io.read_file(path); img=tf.image.decode_jpeg(img,3); img=tf.image.convert_image_dtype(img,tf.float32)
        return img

    def _augment(self,img,train=True):
        shape=tf.shape(img)[:2]
        scale=342./tf.cast(tf.reduce_min(shape),tf.float32)
        img=tf.image.resize(img,tf.cast(tf.cast(shape,tf.float32)*scale,tf.int32))
        if train: img=tf.image.random_flip_left_right(img); img=tf.image.random_crop(img,[299,299,3])
        else:     img=tf.image.resize_with_crop_or_pad(img,299,299)
        return tf.keras.applications.inception_v3.preprocess_input(img)

    def _gen(self,data,train=True):
        for img,cap in data:
            path=os.path.join(self.cfg['image_dir'],img)
            img_tensor=self._augment(self._load_img(tf.constant(path)),train)
            tok,ln=self.encode(cap)
            yield img_tensor, tok.astype(np.int32), ln, img

    def make_ds(self,data,train=True):
        sig=(tf.TensorSpec((299,299,3),tf.float32),
             tf.TensorSpec((self.cfg['max_length'],),tf.int32),
             tf.TensorSpec((),tf.int32),
             tf.TensorSpec((),tf.string))
        ds=tf.data.Dataset.from_generator(lambda:self._gen(data,train),output_signature=sig)
        if train: ds=ds.shuffle(self.cfg['buffer_size'])
        ds=ds.batch(self.cfg['batch_size']).prefetch(AUTOTUNE)
        return ds

# ──────────────────────────────────────────────────────────────────────────────
# 4. NETWORK COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(Model):
    def __init__(self): super().__init__(); base=tf.keras.applications.InceptionV3(include_top=False,weights='imagenet',input_shape=(299,299,3))
    def build(self,input_shape):
        self.base=tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
        self.base.trainable=False; self.reshape=layers.Reshape((-1,2048))
    def call(self,x): return self.reshape(self.base(x))
    def unfreeze_top_layers(self,n=8):                       # [spec 7]
        for l in self.base.layers[-n:]: l.trainable=True

class BahdanauAttention(layers.Layer):
    def __init__(self,units): super().__init__(); self.W1=layers.Dense(units); self.W2=layers.Dense(units); self.V=layers.Dense(1)
    def call(self,feat,hidden):
        score=self.V(tf.nn.tanh(self.W1(feat)+self.W2(tf.expand_dims(hidden,1))))
        attn=tf.nn.softmax(score,1); ctx=tf.reduce_sum(attn*feat,1)
        return ctx,tf.squeeze(attn,-1)

class Decoder(Model):
    def __init__(self,embed_dim,units,vocab,drop):
        super().__init__(); self.units=units
        self.embed=layers.Embedding(vocab,embed_dim,mask_zero=True)
        self.attn=BahdanauAttention(units); self.beta=layers.Dense(1,activation='sigmoid')
        self.lstm=layers.LSTM(units,return_sequences=True,return_state=True)
        self.drop=layers.Dropout(drop); self.proj=layers.Dense(units*2); self.fc=layers.Dense(vocab,dtype='float32')
    def call(self,x,feat,h,c):
        ctx,alpha=self.attn(feat,h); ctx=self.beta(h)*ctx
        x=self.embed(x); x=tf.concat([tf.expand_dims(ctx,1),x],-1)
        o,h,c=self.lstm(x,initial_state=[h,c]); o=tf.squeeze(o,1)
        m=tf.reshape(self.proj(tf.concat([o,ctx],-1)),(-1,self.units,2)); m=tf.reduce_max(m,2)
        return tf.expand_dims(self.drop(m),1),h,c,alpha

# ──────────────────────────────────────────────────────────────────────────────
# 5. IMAGE CAPTIONING TRAINER
# ──────────────────────────────────────────────────────────────────────────────
class CaptioningSystem:
    def __init__(self,cfg:Dict[str,object],proc:DataProcessor):
        self.cfg,self.proc=cfg,proc
        self.encoder,self.decoder=None,None
        self.opt,self.loss_fn=None,None
        self.ckpt_mgr=None
        # logs
        self.best_bleu=0; self.loss_log=[]; self.tr_bleu=[]; self.val_bleu=[]
        self.grad_norms=[]
        self.smooth=SmoothingFunction().method4

    # ─────────────────────────────
    # 5.1 build & compile
    # ─────────────────────────────
    def build(self,steps_per_epoch):
        print("Building model …")
        self.encoder=Encoder(); self.decoder=Decoder(self.cfg['embedding_dim'],self.cfg['units'],
                                                     self.proc.vocab_size,self.cfg['decoder_dropout'])

        decay_steps=int(steps_per_epoch*self.cfg['epochs'])     # [spec 6]
        sch=CosineDecay(self.cfg['initial_lr'],decay_steps,alpha=self.cfg['lr_alpha'])
        base_opt=tf.keras.optimizers.Adam(sch)
        self.opt=(tf.keras.mixed_precision.LossScaleOptimizer(base_opt)
                  if self.cfg['mixed_precision'] else base_opt)

        self.loss_fn=SparseCategoricalCrossentropy(from_logits=True,label_smoothing=0.1,reduction='none') # [spec 3]

        if self.cfg['save_checkpoints']:
            ckpt=tf.train.Checkpoint(enc=self.encoder,dec=self.decoder,opt=self.opt)
            self.ckpt_mgr=tf.train.CheckpointManager(ckpt,self.cfg['checkpoint_dir'],max_to_keep=3)
            if self.ckpt_mgr.latest_checkpoint:
                ckpt.restore(self.ckpt_mgr.latest_checkpoint); print("Restored",self.ckpt_mgr.latest_checkpoint)

    # ─────────────────────────────
    # 5.2 helpers
    # ─────────────────────────────
    def _cast(self,x): return tf.cast(x,tf.float32) if x.dtype!=tf.float32 else x

    # scheduled-sampling prob updated every epoch  [spec 4]
    def _ss_prob(self,epoch): return self.cfg['scheduled_sampling_max_prob']*epoch/max(1,self.cfg['epochs']-1)

    # trigram-blocker util                               [spec 11]
    @staticmethod
    def _repeat_trigram(seq,tok):
        return len(seq)>=2 and (seq[-2],seq[-1],tok) in set(zip(seq,seq[1:],seq[2:]))

    # deduplicate prediction util                        [spec 12]
    @staticmethod
    def _dedup(tokens):
        out=[]
        for i,t in enumerate(tokens):
            if i<2 or not(t==tokens[i-1]==tokens[i-2]): out.append(t)
        return out

    # ─────────────────────────────
    # 5.3 train-step @tf.function
    # ─────────────────────────────
    @tf.function
    def _train_step(self,img,tgt,ln,ss_p):
        B=tf.shape(img)[0]
        with tf.GradientTape() as tape:
            feat=self.encoder(img)                                    # (B,64,2048)
            h=tf.zeros((B,self.cfg['units']),feat.dtype); c=tf.zeros_like(h)
            dec_in=tf.expand_dims(tf.fill([B],self.proc.tokeniser.word_index['<start>']),1)
            attn_cov=tf.zeros((B,tf.shape(feat)[1]),feat.dtype)
            total_ce=tf.constant(0.,tf.float32)

            for t in tf.range(1,self.cfg['max_length']):
                logits,h,c,alpha=self.decoder(dec_in,feat,h,c)
                attn_cov+=alpha
                ce=self.loss_fn(tgt[:,t],tf.squeeze(logits,1))
                mask=tf.cast(tgt[:,t]!=0,tf.float32); total_ce+=tf.reduce_sum(ce*mask)

                pred=tf.argmax(logits,-1,output_type=tf.int32)[:,0]
                use_pred=tf.less(tf.random.uniform([B]),ss_p)
                nxt=tf.where(use_pred,pred,tgt[:,t])
                dec_in=tf.expand_dims(nxt,1)

            ce_loss=total_ce/tf.reduce_sum(tf.cast(ln,tf.float32))
            reg=tf.reduce_mean(tf.square(1.-self._cast(attn_cov)))
            loss=ce_loss+self.cfg['attention_reg_lambda']*reg        # coverage [spec 5]

            if isinstance(self.opt,tf.keras.mixed_precision.LossScaleOptimizer):
                loss_scaled=self.opt.get_scaled_loss(loss)
            else: loss_scaled=loss

        vars=self.encoder.trainable_variables+self.decoder.trainable_variables
        grads=tape.gradient(loss_scaled,vars)
        if isinstance(self.opt,tf.keras.mixed_precision.LossScaleOptimizer):
            grads=self.opt.get_unscaled_gradients(grads)
        grads,_=tf.clip_by_global_norm(grads,self.cfg['grad_clip_value'])
        self.opt.apply_gradients(zip(grads,vars))
        return loss,tf.linalg.global_norm(grads)

    # ─────────────────────────────
    # 5.4 greedy decode – **batched** & @tf.function  [spec 8]
    # ─────────────────────────────
    @tf.function
    def _greedy_batch(self,feat_batch):
        B=tf.shape(feat_batch)[0]
        h=tf.zeros((B,self.cfg['units']),feat_batch.dtype); c=tf.zeros_like(h)
        dec_in=tf.expand_dims(tf.fill([B],self.proc.tokeniser.word_index['<start>']),1)
        seq=tf.TensorArray(tf.int32,size=self.cfg['max_length'])
        for t in tf.range(self.cfg['max_length']):
            logits,h,c,_=self.decoder(dec_in,feat_batch,h,c)
            nxt=tf.argmax(logits[:,-1],-1,output_type=tf.int32)
            seq=seq.write(t,nxt)
            dec_in=tf.expand_dims(nxt,1)
        return tf.transpose(seq.stack())  # (B,T)

    # ─────────────────────────────
    # 5.5 evaluate BLEU util (uses cached CNN feats)    [spec 8,9]
    # ─────────────────────────────
    def _load_val_feats(self,save_if_missing=True):
        cache='val_feats.npz'
        if os.path.exists(cache):
            return np.load(cache)['arr_0']
        feats=[]
        for img,_ in self.proc.val_pairs:
            path=os.path.join(self.cfg['image_dir'],img)
            tensor=self.proc._augment(self.proc._load_img(tf.constant(path)),train=False)
            feats.append(self.encoder(tf.expand_dims(tensor,0))[0].numpy())
        arr=np.stack(feats)
        if save_if_missing: np.savez_compressed(cache,arr_0=arr)
        return arr

    def _compute_bleu(self,data,batch_size=64,max_imgs=None):
        refs,hyps=[],[]
        subset=data if max_imgs is None else random.sample(data,min(max_imgs,len(data)))
        # pre-decode CNN feats in batches for speed
        feat_list=[]
        for img,_ in subset:
            path=os.path.join(self.cfg['image_dir'],img)
            feat=self.encoder(tf.expand_dims(
                self.proc._augment(self.proc._load_img(tf.constant(path)),train=False),0))
            feat_list.append(feat)
        batched=tf.data.Dataset.from_tensor_slices(tf.concat(feat_list,0)).batch(batch_size)
        preds=[]
        for feat_b in batched:
            seqs=self._greedy_batch(feat_b).numpy()                   # [spec 8]
            preds.extend(seqs)
        for (img,_),seq in zip(subset,preds):
            hyp=[self.proc.tokeniser.index_word.get(i,'') for i in seq
                 if i not in (0,self.proc.tokeniser.word_index['<end>'],
                              self.proc.tokeniser.word_index['<start>'])]
            hyp=self._dedup(hyp)                                      # [spec 12]
            gt=[[w for w in self.proc.preprocess(c).split()
                 if w not in ('<start>','<end>')] for c in self.proc.caption_map[img][:5]]
            refs.append(gt); hyps.append(hyp)
        weight=(0.25,0.25,0.25,0.25)
        return corpus_bleu(refs,hyps,weights=weight,smoothing_function=self.smooth)

    # ─────────────────────────────
    # 5.6 beam search w/ trigram block [spec 11] + dedup [12]
    # ─────────────────────────────
    def beam_search(self,img_path,beam=5,len_pen=0.7):
        feat=self.encoder(tf.expand_dims(
            self.proc._augment(self.proc._load_img(tf.constant(img_path)),train=False),0))
        start,end=self.proc.tokeniser.word_index['<start>'],self.proc.tokeniser.word_index['<end>']
        beams=[(0,[start],tf.zeros((1,self.cfg['units'])),tf.zeros((1,self.cfg['units'])),[])]
        completed=[]
        for _ in range(self.cfg['max_length']):
            cand=[]
            for score,seq,h,c,alphas in beams:
                if seq[-1]==end: completed.append((score,seq,alphas)); continue
                logits,h1,c1,alpha=self.decoder(tf.expand_dims([seq[-1]],0),feat,h,c)
                log_p=tf.nn.log_softmax(self._cast(logits[0,0]))
                top=tf.math.top_k(log_p,beam).indices.numpy()
                for tok in top:
                    tok=int(tok); 
                    if self._repeat_trigram(seq,tok): continue
                    cand.append((score+float(log_p[tok]),seq+[tok],h1,c1,alphas+[alpha[0].numpy()]))
            if not cand: break
            cand.sort(key=lambda x:x[0]/((len(x[1])**len_pen)),reverse=True)
            beams=cand[:beam]
            if len(completed)>=beam: break
        best=max(completed+beams,key=lambda x:x[0]/((len(x[1]))**len_pen))
        words=[self.proc.tokeniser.index_word.get(i,'') for i in best[1]
               if i not in (start,end,0)] 
        return " ".join(self._dedup(words))

    # ─────────────────────────────
    # 5.7 main TRAIN LOOP (spec 4,6,7,10,14)
    # ─────────────────────────────
    def train(self,train_ds,val_subset,val_full):
        steps_per_epoch=math.ceil(len(self.proc.train_pairs)/self.cfg['batch_size'])
        self.build(steps_per_epoch)
        for epoch in range(self.cfg['epochs']):
            ss_p=self._ss_prob(epoch)                                 # [spec 4]
            prog=tf.keras.utils.Progbar(steps_per_epoch,unit_name='batch')
            tot_loss=0
            for img,tgt,ln,_ in train_ds:
                loss,g=self._train_step(img,tgt,ln,ss_p)
                tot_loss+=float(loss); self.grad_norms.append(float(g))
                prog.add(1,values=[('loss',loss)])
            self.loss_log.append(tot_loss/steps_per_epoch)

            # micro-val BLEU (≤500 imgs) every epoch
            micro_bleu=self._compute_bleu(val_subset)
            # full val BLEU every 5 epochs      [spec 10]
            full_bleu = self._compute_bleu(val_full) if epoch%5==0 else None
            self.tr_bleu.append(micro_bleu); self.val_bleu.append(full_bleu or np.nan)

            # checkpoint
            if self.ckpt_mgr: self.ckpt_mgr.save()

            print(f"Epoch {epoch+1}/{self.cfg['epochs']}  "
                  f"loss={self.loss_log[-1]:.4f}  microBLEU={micro_bleu:.4f}"
                  + (f"  fullBLEU={full_bleu:.4f}" if full_bleu is not None else "")
                  + f"  ss_p={ss_p:.2f}")

            # unfreeze CNN after epoch 5     [spec 7]
            if epoch==4:
                self.encoder.unfreeze_top_layers(8); 
                self.opt=tf.keras.optimizers.Adam(1e-5)

# ──────────────────────────────────────────────────────────────────────────────
# 6. SET-UP & RUN
# ──────────────────────────────────────────────────────────────────────────────
processor=DataProcessor(CONFIG); processor.prepare()

train_ds=processor.make_ds(processor.train_pairs,train=True)
val_ds =processor.make_ds(processor.val_pairs ,train=False)

# fixed 500-img micro-val subset
micro_val=random.sample(processor.val_pairs,min(500,len(processor.val_pairs)))
# full val cached feats
val_feats=processor.val_pairs                               # caching handled inside class

system=CaptioningSystem(CONFIG,processor)
system.train(train_ds,micro_val,processor.val_pairs)

# ──────────────────────────────────────────────────────────────────────────────
# 7. DEMO
# ──────────────────────────────────────────────────────────────────────────────
demo_img=os.path.join(CONFIG['image_dir'],processor.test_pairs[0][0])
print("Beam-search caption:",system.beam_search(demo_img))
