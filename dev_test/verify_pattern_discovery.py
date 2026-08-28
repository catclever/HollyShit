import sys
import os
sys.path.insert(0, '/Users/kael/workbench/HollyShit')

import mlx.core as mx
import numpy as np
import pandas as pd
from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from tools.weight_utils import load_torch_weights_into_mlx

tokenizer = CharTokenizer()

def chunk_h(h_np, tokens, threshold=0.8):
    L = h_np.shape[0]
    norms = np.linalg.norm(h_np, ord=2, axis=-1, keepdims=True)
    h_norm = h_np / np.maximum(norms, 1e-8)
    
    punctuations = set('。，、；：！？\n ()（）《》「」\"\'')
    
    chunks = []
    chunk_start = 0
    current_sum = h_np[0].copy()
    current_count = 1
    current_mean_norm = h_norm[0].copy()
    
    for i in range(1, L):
        sim = np.dot(h_norm[i], current_mean_norm)
        char = tokenizer.decode([tokens[i]])
        prev_char = tokenizer.decode([tokens[i-1]])
        is_punct = char in punctuations
        was_punct = prev_char in punctuations
        
        if sim > threshold and not is_punct and not was_punct:
            current_sum += h_np[i]
            current_count += 1
            current_mean = current_sum / current_count
            current_mean_norm = current_mean / max(np.linalg.norm(current_mean), 1e-8)
        else:
            chunks.append({'mean': current_sum / current_count, 'len': current_count, 'start': chunk_start, 'end': i})
            chunk_start = i
            current_sum = h_np[i].copy()
            current_count = 1
            current_mean_norm = h_norm[i].copy()
            
    chunks.append({'mean': current_sum / current_count, 'len': current_count, 'start': chunk_start, 'end': L})
    return chunks

encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
load_torch_weights_into_mlx(encoder, 'checkpoints/basencoder_step_115000/student.pt')
mx.eval(encoder.parameters())

@mx.compile
def encode_text(tokens_mx):
    h = encoder.get_normalized_embeddings(tokens_mx)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)

df = pd.read_parquet('data/Basic_ZH/wiki_filtered.parquet')

# Test on 5 different texts
count = 0
for t in df['completion'].dropna():
    if len(t) > 300 and len(t) < 800:
        text = t
        tokens = tokenizer.encode(text)
        tokens_mx = mx.array([tokens], dtype=mx.int32)
        h = np.array(encode_text(tokens_mx))[0]

        chunks = chunk_h(h, tokens, threshold=0.8)
        
        C = np.array([c['mean'] for c in chunks])
        C_norms = np.linalg.norm(C, ord=2, axis=-1, keepdims=True)
        C_norm = C / np.maximum(C_norms, 1e-8)
        S = np.dot(C_norm, C_norm.T)

        window = 5
        N = len(chunks)
        mask = np.ones_like(S)
        for i in range(N):
            start = max(0, i - window)
            end = min(N, i + window + 1)
            mask[i, start:end] = 0.0

        S_masked = S * mask
        resonance_thresh = 0.90
        recurrence_scores = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if S_masked[i, j] > resonance_thresh:
                    recurrence_scores[i] += chunks[j]['len'] * chunks[i]['len'] 

        top_indices = np.argsort(recurrence_scores)[::-1]
        
        print(f"\n=====================================")
        print(f"DOCUMENT {count+1} (Length: {len(text)} chars)")
        print(f"Preview: {text[:150].replace(chr(10), ' ')}...")
        print(f"--- Top 5 Resonant Concepts ---")
        
        seen_texts = set()
        out_count = 0
        for idx in top_indices:
            if recurrence_scores[idx] < 1:
                break
            
            c = chunks[idx]
            text_segment = tokenizer.decode(tokens[c['start']:c['end']]).replace('\n', ' ')
            
            if text_segment in ['的', '了', '在', '是', '和', '与', '、', '，', '。', '（', '）', ' ', '于', '有', '为'] or len(text_segment.strip()) == 0:
                continue
                
            if text_segment not in seen_texts:
                seen_texts.add(text_segment)
                print(f"[{out_count+1}] '{text_segment}' (Len: {c['len']}) | Score: {recurrence_scores[idx]}")
                out_count += 1
                if out_count >= 5:
                    break
                    
        count += 1
        if count >= 5:
            break
