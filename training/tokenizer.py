import collections
import re
import json

class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        
    # ... (the rest of your methods like get_stats, merge, train, encode, decode are fine) ...

    def get_stats(self, ids):
        counts = collections.defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        words = re.findall(r'\w+|\S', text)
        ids = list(range(256))
        
        char_vocab = sorted(list(set(''.join(words))))
        char_to_id = {char: i for i, char in enumerate(char_vocab)}
        
        tokens = [[char_to_id[c] for c in word] for word in words]
        self.vocab = {i: chr(i) for i in range(256)}
        
        for i in range(self.vocab_size - len(self.vocab)):
            stats = collections.defaultdict(int)
            for token_ids in tokens:
                stats.update(self.get_stats(token_ids))

            if not stats:
                break
            
            best_pair = max(stats, key=stats.get)
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_tokens = []
            for token_ids in tokens:
                new_tokens.append(self.merge(token_ids, best_pair, new_id))
            tokens = new_tokens
    
    def encode(self, text):
        tokens = []
        for char in text:
            tokens.append(ord(char))
        
        while True:
            stats = self.get_stats(tokens)
            pair = max(self.merges, key=self.merges.get, default=None)
            if pair is None or pair not in stats:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
    
    def decode(self, tokens):
        text = ''.join([self.vocab[i] for i in tokens])
        return text

    # --- FIXES BELOW ---
    def save(self, file_path):
        # Convert tuple keys to strings before saving
        str_merges = {str(k): v for k, v in self.merges.items()}
        # vocab keys are already ints, which is fine, but we'll convert them just to be safe
        str_vocab = {str(k): v for k, v in self.vocab.items()}
        
        with open(file_path, 'w') as f:
            json.dump({'merges': str_merges, 'vocab': str_vocab}, f)

    @classmethod
    def load(cls, file_path):
        tokenizer = cls()
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Convert string keys back to tuples for merges
            tokenizer.merges = {eval(k): v for k, v in data['merges'].items()}
            # Convert string keys back to ints for vocab
            tokenizer.vocab = {int(k): v for k, v in data['vocab'].items()}
        return tokenizer
    # --- END OF FIXES ---


if __name__ == '__main__':
    # Make sure this path is correct for your project structure
    # Based on your previous commands, `data/nuck_dataset.txt` should be right
    try:
        with open('./data/nuck_dataset.txt', 'r', encoding='utf-8') as f:
            text_data = f.read()

        tokenizer = SimpleTokenizer(vocab_size=1000)
        tokenizer.train(text_data)
        tokenizer.save('nuck_tokenizer.json')

        print("Tokenizer trained and saved to nuck_tokenizer.json")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")

    except FileNotFoundError:
        print("Please make sure 'data/nuck_dataset.txt' exists in your NuckAI directory.")