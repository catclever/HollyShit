import json

class CharTokenizer:
    def __init__(self, vocab_path="data/Basic_ZH/char_vocab.json"):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]
        self.bos_token_id = self.vocab["<BOS>"]
        self.eos_token_id = self.vocab["<EOS>"]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        """
        Converts a string of text into a sequence of token IDs.
        ASCII and unknown characters fallback to 0-255 foundational bytes.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
            
        for char in text:
            if char in self.vocab:
                ids.append(self.vocab[char])
            else:
                # UTF-8 Byte Fallback!
                # Break the character down into 1-4 raw bytes.
                # Because 0-255 are strictly mapped to IDs 0-255, we just append the byte value!
                encoded_bytes = char.encode('utf-8')
                for b in encoded_bytes:
                    ids.append(b)
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
            
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True):
        """
        Reconstructs the original string from a sequence of IDs.
        Automatically catches and decodes UTF-8 byte sequences gracefully.
        """
        special = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
        
        byte_buffer = bytearray()
        result = ""
        
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            
            # If it's a raw byte (0-255), add it to the byte accumulation buffer
            if 0 <= i <= 255:
                byte_buffer.append(i)
            else:
                # Flush the byte buffer if we hit a standard character ID
                if byte_buffer:
                    try:
                        result += byte_buffer.decode('utf-8')
                    except UnicodeDecodeError:
                        result += "" # Standard replacement character for broken sequences
                    byte_buffer.clear()
                    
                char_str = self.id_to_char.get(i, "")
                if char_str not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                    result += char_str
                    
        # Flush whatever bytes are left at the end
        if byte_buffer:
            try:
                result += byte_buffer.decode('utf-8')
            except UnicodeDecodeError:
                result += ""
                
        return result
