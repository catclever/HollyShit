import pandas as pd
import numpy as np
import mlx.core as mx
import spacy
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from tools.weight_utils import load_torch_weights_into_mlx

def compute_features(mx_c):
    # Returns magnitude (L2), diffs (Gradient sharpness)
    mag = mx.linalg.norm(mx_c, axis=-1)
    # diffs[i] = ||c[i+1] - c[i]||_1
    diffs = mx.sum(mx.abs(mx_c[:, 1:] - mx_c[:, :-1]), axis=-1)
    # Pad diffs to match length
    diffs = mx.concatenate([mx.zeros((mx_c.shape[0], 1)), diffs], axis=1)
    return mag, diffs

def main():
    print("Loading SpaCy zh_core_web_sm...")
    nlp = spacy.load("zh_core_web_sm")
    
    # Load dataset
    print("Loading Dataset...")
    df = pd.read_parquet("data/Basic_ZH/wiki_filtered.parquet").head(1000) # Sample 1000
    texts = df['completion'].tolist()
    
    # Load Model
    print("Loading Tokenizer and Model...")
    tokenizer = CharTokenizer("data/Basic_ZH/char_vocab.json")
    encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
    load_torch_weights_into_mlx(encoder, "checkpoints/basencoder_step_115000/student.pt")
    encoder.eval()
    
    @mx.compile
    def encode_batch(x):
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
        h = encoder.final_ln(h)
        return h
        
    entity_features = {'mag': [], 'diff': []}
    non_entity_features = {'mag': [], 'diff': []}
    
    # NER Labels to track
    target_labels = {'PERSON', 'LOC', 'ORG', 'GPE', 'FAC'}
    
    print("Starting processing...")
    for text in tqdm(texts):
        # We need to process in chunks if text is long
        text = text[:512]
        if not text.strip(): continue
        
        # 1. SpaCy NER
        doc = nlp(text)
        is_entity = [0] * len(text)
        for ent in doc.ents:
            if ent.label_ in target_labels:
                for i in range(ent.start_char, ent.end_char):
                    if i < len(is_entity):
                        is_entity[i] = 1
                        
        # 2. Tokenize and align
        tokens = tokenizer.encode(text, add_special_tokens=False)
        mx_x = mx.array([tokens])
        h = encode_batch(mx_x)
        mag, diff = compute_features(h)
        mag = np.array(mag)[0]
        diff = np.array(diff)[0]
        
        # We need to align characters to tokens (since TinyCharEncoder is character-based, it's roughly 1:1, but we should be careful with UNK)
        # For simplicity in this experiment, we assume 1:1 matching for valid chars
        char_idx = 0
        for i, token in enumerate(tokens):
            if char_idx >= len(is_entity): break
            # Decode token to see how many chars it consumes
            char_str = tokenizer.decode([token])
            
            ent_flag = is_entity[char_idx]
            
            if ent_flag == 1:
                entity_features['mag'].append(mag[i].item())
                entity_features['diff'].append(diff[i].item())
            else:
                non_entity_features['mag'].append(mag[i].item())
                non_entity_features['diff'].append(diff[i].item())
                
            char_idx += len(char_str)
            
    print("\n--- Statistics ---")
    ent_mag = np.array(entity_features['mag'])
    non_ent_mag = np.array(non_entity_features['mag'])
    ent_diff = np.array(entity_features['diff'])
    non_ent_diff = np.array(non_entity_features['diff'])
    
    print(f"Entities Count: {len(ent_mag)}")
    print(f"Non-Entities Count: {len(non_ent_mag)}")
    
    print(f"\n[L2 Norm Magnitude]")
    print(f"Entity:     Mean = {ent_mag.mean():.4f}, Std = {ent_mag.std():.4f}")
    print(f"Non-Entity: Mean = {non_ent_mag.mean():.4f}, Std = {non_ent_mag.std():.4f}")
    
    print(f"\n[Boundary Diff]")
    print(f"Entity:     Mean = {ent_diff.mean():.4f}, Std = {ent_diff.std():.4f}")
    print(f"Non-Entity: Mean = {non_ent_diff.mean():.4f}, Std = {non_ent_diff.std():.4f}")

if __name__ == '__main__':
    main()
