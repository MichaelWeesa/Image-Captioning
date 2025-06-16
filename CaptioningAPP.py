import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization
import streamlit as st
import tempfile
from PIL import Image
import cv2
import requests
import json

# ---- PARAMETERS ----
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
IMAGE_SIZE = (224, 224)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "").replace(">", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# ---- LOAD VECTORIZE ----
def load_vocab(filename):
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f]
vocab = load_vocab("my_vocab.txt")
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.set_vocabulary(vocab)

# ---- MODEL DEFINITION ----
def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            combined_mask = causal_mask
            padding_mask = None
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model, encoder, decoder, image_aug=None,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

    def call(self, inputs, training=False):
        images, sequences = inputs
        img_embed = self.cnn_model(images)
        encoder_out = self.encoder(img_embed, training=training)
        output = self.decoder(sequences, encoder_out, training=training)
        return output

# ---- INSTANTIATE MODEL AND LOAD WEIGHTS ----
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=None,
)
dummy_images = tf.random.uniform((1, *IMAGE_SIZE, 3))
dummy_sequences = tf.random.uniform((1, SEQ_LENGTH), maxval=VOCAB_SIZE, dtype=tf.int32)
_ = caption_model((dummy_images, dummy_sequences), training=True)
WEIGHTS_PATH = 'best_transformer_1_weights.h5'
caption_model.load_weights(WEIGHTS_PATH)

# ---- BEAM SEARCH ----
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

def beam_search_caption(image_path, beam_width=5, alpha=20, max_length=25):
    sample_img = decode_and_resize(image_path)
    img = tf.expand_dims(sample_img, 0)
    img_features = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img_features, training=False)
    start_token = vectorization('<start>').numpy()[0]
    end_token = vectorization('<end>').numpy()[0]
    beam = [(0.0, [start_token])]

    import heapq
    for _ in range(max_length):
        candidates = []
        for score, seq in beam:
            if seq[-1] == end_token:
                candidates.append((score, seq))
                continue

            seq_input = tf.expand_dims(seq, 0)
            mask = tf.math.not_equal(seq_input, 0)
            preds = caption_model.decoder(seq_input, encoded_img, training=False, mask=mask)
            preds = preds[0, -1, :].numpy()
            top_indices = np.argsort(preds)[-beam_width:]
            for idx in top_indices:
                prob = preds[idx]
                if prob == 0:
                    continue
                new_seq = seq + [idx]
                length_penalty = ((5 + len(new_seq)) / 6) ** alpha
                new_score = score + np.log(prob) / length_penalty
                candidates.append((new_score, new_seq))

        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        if all(seq[-1] == end_token for _, seq in beam):
            break

    _, best_seq = max(beam, key=lambda x: x[0])
    tokens = [index_lookup.get(idx, '') for idx in best_seq]
    caption_tokens = [t for t in tokens if t not in ['<start>', '<end>', '']]
    caption = ' '.join(caption_tokens).strip()
    return caption

# ---- VIDEO CAPTIONING HELPERS ----
def extract_video_frames(video_path, frame_interval=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames

def summarize_captions_with_openrouter(joined_captions_text):
    prompt = (
        "Summarize the following captions from a video into one natural-sounding sentence. ONLY RETURN THE CAPTION AS YOUR RESPONSE: \n"
        f"{joined_captions_text}"
    )

    headers = {
        "Authorization": "Bearer YOUR API KEY FROM OPEN ROUTER",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "YOUR TITLE"
    }

    payload = {
        "model": "openai/gpt-4.1-nano",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "[Error generating summary]"

def caption_video(video_path, beam_width=5, alpha=20, frame_interval=60, show_frames=False, st_container=None):
    frames = extract_video_frames(video_path, frame_interval=frame_interval)
    all_captions = []
    n_frames = len(frames)
    if n_frames == 0:
        return "[No frames extracted]", "[No summary]"

    progress_bar = None
    if st_container is not None:
        progress_bar = st_container.progress(0, text="Generating Captions...")

    # --- Prepare containers for immediate, grid-like display ---
    image_slots = []
    if show_frames and st_container is not None:
        n_rows = (n_frames + 2) // 3
        for _ in range(n_rows):
            image_slots.append(st_container.columns(3))

    for idx, frame in enumerate(frames):
        temp_img_path = f"temp_frame_{idx}.jpg"
        cv2.imwrite(temp_img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        caption = beam_search_caption(temp_img_path, beam_width=beam_width, alpha=alpha)
        all_captions.append(caption)

        # --- Show each frame in its cell as soon as it's ready ---
        if show_frames and st_container is not None:
            row = idx // 3
            col = idx % 3
            image_slots[row][col].image(frame, caption=f"Frame {idx+1}: {caption}", use_column_width=True)

        os.remove(temp_img_path)

        if progress_bar is not None:
            progress_bar.progress((idx + 1) / n_frames, text=f"Captioned frame {idx+1}/{n_frames}")

    joined_caption_text = ". ".join(all_captions) + "."
    summary = summarize_captions_with_openrouter(joined_caption_text)
    return joined_caption_text, summary

# ---- STREAMLIT APP ----
st.title("🧠 CNN-Transformer Captioning")
mode = st.radio("Choose mode:", ["📷 Image Captioning", "🎥 Video Captioning"], horizontal=True)
beam_width = 5
alpha = 20
max_length = 25

if mode == "📷 Image Captioning":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile_path = tmpfile.name

        caption = beam_search_caption(tmpfile_path, beam_width=beam_width, alpha=alpha, max_length=max_length)

        img = Image.open(tmpfile_path).convert("RGB")
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.markdown(f"**📝 Caption:** {caption}")

elif mode == "🎥 Video Captioning":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    frame_interval = st.number_input("Extract one frame every N frames", min_value=1, max_value=240, value=60, step=1)

    show_frames = st.checkbox("Show all frames & captions", value=False)
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpvideo:
            tmpvideo.write(uploaded_video.read())
            tmpvideo_path = tmpvideo.name

        video_container = st.container()
        joined_captions, summary = caption_video(
            tmpvideo_path,
            beam_width=beam_width,
            alpha=alpha,
            frame_interval=frame_interval,
            show_frames=show_frames,
            st_container=video_container
        )

        tab1, tab2 = st.tabs(["🧠 Summary", "🖼️ Captions"])
        with tab1:
            st.markdown("#### 🤖 Summary by Language Model")
            st.write(summary)
        with tab2:
            st.markdown("#### 📷 Captions from Frames")
            st.write(joined_captions)
