import pandas as pd
import re

def intelligent_chunking(text):
    """
    Chunks text based on punctuation and newlines, handling EOS tokens carefully.
    
    Rules:
    1. Split text into chunks at `。`, `！`, `？`, and `\n`.
    2. Punctuation marks must be kept with the chunk they follow.
    3. An `<EOS>` token is ONLY added when a `\n` is encountered, OR at the very end of the text.
    4. If multiple split characters occur consecutively (e.g., "。\n"), they are all appended to the preceding chunk.
    5. The `<EOS>` token (if applicable based on rule 3) is appended to the chunk text itself (e.g. "你好。\n<EOS>").
    
    Returns:
        list of str: The segmented chunks with appropriate punctuation and EOS appended.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Regex to find all text chunks and their immediately following delimiters
    # Matches any non-delimiter characters followed by any number of delimiters (。\n！？!?)
    # or just the remaining non-delimiter text at the end.
    pattern = re.compile(r'([^。！？!?\n]+)([。！？!?\n]*)')
    
    chunks = []
    
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        delimiters = match.group(2)
        
        if not content and not delimiters:
            continue
            
        chunk_text = content + delimiters
        
        # Determine if we need an EOS
        # We need an EOS if there is a newline in the delimiters
        if '\n' in delimiters:
            chunk_text = chunk_text.replace('\n', '') + '<EOS>'
            
        chunks.append(chunk_text)
        
    # The absolute last chunk of any text (like a continuous Wiki article) 
    # must always have an EOS if it doesn't already have one from a trailing newline.
    if chunks and not chunks[-1].endswith('<EOS>'):
        chunks[-1] += '<EOS>'
        
    # Clean up any purely empty chunks that might have sneaked in
    chunks = [c.strip() for c in chunks if c.strip() and c.strip() != '<EOS>']

    return chunks

def main():
    input_path = 'data/Basic_ZH/cleaned_mixed_wiki.parquet'
    output_path = 'data/Basic_ZH/chunked_mixed_wiki.parquet'
    
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.")
    
    print("Applying intelligent chunking and EOS injection...")
    df['chunks'] = df['text'].apply(intelligent_chunking)
    df['chunk_count'] = df['chunks'].apply(len)
    
    # Filter valid rows
    valid_df = df[df['chunk_count'] > 0].copy()
    valid_df = valid_df.drop(columns=['text'])
    
    print(f"Finished processing. Valid rows: {len(valid_df)}")
    print(f"Average chunks per row: {valid_df['chunk_count'].mean():.2f}")
    
    print("\n--- Sample of chunked data ---")
    for i, row in valid_df.head(5).iterrows():
        print(f"\nRow {i} ({row['chunk_count']} chunks):")
        for j, chunk in enumerate(row['chunks'][:5]):
            print(f"  [{j}]: {repr(chunk)}")
        if row['chunk_count'] > 5:
            print("  ...")
            
    print(f"\nSaving to {output_path}...")
    valid_df.to_parquet(output_path, engine='pyarrow', index=False)
    print("Done!")

if __name__ == '__main__':
    main()
