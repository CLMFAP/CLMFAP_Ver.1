import regex as re
import torch
import numpy as np
import random
import collections


class Encoder:

    def __init__(self, max_length=50, add_bos=False, add_eos=False, feature_size=32, mlm_probability = 0.15):
        self.vocab_encoder = torch.load("dictionary.pth")

        self.max_length = max_length
        self.min_length = 0
        self.mod_length = 1000
        self.mlm_probability = mlm_probability
        self.avg_length = 66
        self.tail = 122
        self.b0_cache = collections.deque()
        self.b1_cache = collections.deque()
        self.b2_cache = collections.deque()
        self.b3_cache = collections.deque()
        self.bucket0 = collections.deque()
        self.bucket1 = collections.deque()
        self.bucket2 = collections.deque()
        self.bucket3 = collections.deque()
        if feature_size == 32:
            self.b0_max = 100000
            self.b1_max = 700
            self.b2_max = 150
            self.b3_max = 50
        else:
            self.b0_max = 1382
            self.b1_max = 871
            self.b2_max = 516
            self.b3_max = 311
        values = list(self.vocab_encoder.values())
        num_top = 0
        middle_top = 0
        bottom = 0
        for count in values:
            if count > 100000:
                num_top += 1
            if count > 50:
                middle_top += 1
        middle_top = middle_top - num_top
        self.cutoffs = [num_top + 4, middle_top]
        self.char2id = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<mask>": 3}
        self.id2char = {0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<mask>"}
        self.pad = self.char2id["<pad>"]
        self.mask = self.char2id["<mask>"]
        self.eos = self.char2id["<eos>"]
        self.bos = self.char2id["<bos>"]
        pos = 0
        for key, value in self.vocab_encoder.items():
            # for pos, key in enumerate(self.vocab_encoder.keys()):
            self.char2id[key] = pos + 4
            self.id2char[pos + 4] = key
            pos += 1
        self.char2id["<unk>"] = pos + 4
        self.id2char[pos + 4] = "<unk>"
        self.pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.add_bos = add_bos
        self.add_eos = add_eos

    def encode(self, char):
        # if len(char) > self.max_length:
        #    char = char[:self.max_length]
        if self.add_bos == True:
            char = ["<bos>"] + char
        if self.add_eos == True:
            char = char + ["<eos>"]

        return torch.tensor([self.char2id[word] for word in char])

    def encoder(self, tokens):
        # return *map(lambda x: self.encode(x), tokens)
        return [self.encode(mol) for mol in tokens]
    
    def quantize_vector(self, encoded_seq, num_tokens):
        """
        Quantize the float vector to nearest integer index within [0, num_tokens-1].
        Assumes vector values are normalized or can be scaled to fit this range.
        """
        # Example scaling, adjust depending on data range
        threshold = 1e-5

        vector = []
        for value in encoded_seq:
            if abs(value) < threshold:
                break  # Stop once a number below the threshold is encountered
            vector.append(value)

        # print(vector)
        # vector = [value for value in vector if abs(value) > threshold]
        # vector = np.clip(vector, -100, 100)  # Clip values within a range
        # vector = (vector + 100) / 200 * (num_tokens - 1)  # Normalize and scale
        # indices = np.round(vector).astype(int)
        # indices = np.clip(indices, 0, num_tokens - 1)  # Ensure indices stay within bounds
        # return indices
        return vector
    
    def scale_vector(self, vector, num_bins=40):
        """
        Scale the vector to match a target range more closely. This example scales 
        data into discrete bins resembling your expected output.
        """
        vector = np.array(vector)

        # Define binning range based on min/max of vector
        min_val = np.min(vector)
        max_val = np.max(vector)
        bin_edges = np.linspace(min_val, max_val, num_bins)  # 40 bins
        
        # Digitize to map each value to the nearest bin
        binned_vector = np.digitize(vector, bin_edges, right=True)

        # Shift by expected minimum output (e.g., '10' as the starting point)
        scaled_vector = binned_vector + 10 - binned_vector[0]
        
        # Clip final values for consistency within bounds
        scaled_vector = np.clip(scaled_vector, 0, num_bins - 1)

        return scaled_vector.tolist()
        
    def decode(self, encoded_seq):
        """
        Decodes a sequence of token IDs back to the original sentence.
        
        Args:
            encoded_seq (torch.Tensor): A tensor of token IDs.
            
        Returns:
            str: The decoded sentence as a single string.
        """
        # print(len(self.id2char))
        encoded_seq = self.quantize_vector(encoded_seq, num_tokens = 64)
        # print(encoded_seq)

        encoded_seq = self.scale_vector(encoded_seq)

        # print(encoded_seq)
        # Convert token IDs back to characters using the id2char dictionary
        decoded_chars = [self.id2char[token_id] for token_id in encoded_seq]
        
        # Remove special tokens if present
        # if self.add_bos and decoded_chars[0] == "<bos>":
        #     decoded_chars = decoded_chars[1:]  # Remove <bos> token
        # if self.add_eos and decoded_chars[-1] == "<eos>":
        #     decoded_chars = decoded_chars[:-1]  # Remove <eos> token
        # Remove all instances of special tokens if present
        special_tokens = {"<bos>","<mask>","<pad>", "<eos>"}
        decoded_chars = [char for char in decoded_chars if char not in special_tokens]
        
        # Join characters into a single string
        decoded_sentence = ''.join(decoded_chars)
        
        return decoded_sentence

    def process_text(self, text):
        # print(text)
        # random length sequences seems to help training
        mod_length = self.mod_length  # + random.randint(-1, 3)
        avg_length = self.avg_length  # + random.randint(-3, 5)
        # print(f"len text {len(text)}")
        for mol in text:
            # fill up buckets and caches
            # print(f" mol {mol}")
            if "\n" in mol:
                print("carriage return in mol")
            raw_regex = self.regex.findall(mol.strip("\n"))
            length = len(raw_regex)
            if length > self.min_length and length < mod_length:
                if len(self.bucket0) < self.b0_max:
                    self.bucket0.append(raw_regex)
                else:
                    self.b0_cache.append(raw_regex)
            elif length >= mod_length and length < avg_length:
                if len(self.bucket1) < self.b1_max:
                    self.bucket1.append(raw_regex)
                else:
                    self.b1_cache.append(raw_regex)
            elif length >= avg_length and length < self.tail:
                self.b2_cache.append(raw_regex)
                # if len(bucket2) < self.b2_max:
                #    bucket2.append(raw_regex)
                # else:
                #    self.b2_cache.append(raw_regex)
            elif length >= self.tail and length < self.max_length:
                self.b3_cache.append(raw_regex)
            # if len(bucket3) < self.b3_max:
            #     bucket3.append(raw_regex)
            # else:
            #     print("$$$$$$$$$$$$$$$$")
            #     print(mol)

        # print(f"len bucket0 {len(self.bucket0)}")
        # print('before Cache size  {} {} {} {}'.format(len(self.b0_cache), len(self.b1_cache), len(self.b2_cache), len(self.b3_cache)))
        # pour cache elements into any open bucket
        if len(self.bucket0) < self.b0_max and len(self.b0_cache) > 0:
            cache_size = len(self.b0_cache)
            max_margin = self.b0_max - len(self.bucket0)
            range0 = min(cache_size, max_margin)
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))] + [
                self.b0_cache.pop() for i in range(range0)
            ]
            # self.b0_cache =  collections.deque(self.b0_cache[:self.b0_max-len(bucket0)])
            # print('0 type {}'.format(type(self.b0_cache)))
        else:
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))]
        if len(self.bucket1) < self.b1_max and len(self.b1_cache) > 0:
            cache_size = len(self.b1_cache)
            max_margin = self.b1_max - len(self.bucket1)
            range1 = min(cache_size, max_margin)
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))] + [
                self.b1_cache.pop() for i in range(range1)
            ]
        else:
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))]

        if len(self.b2_cache) > self.b2_max:
            cache_size = len(self.b2_cache)
            max_margin = self.b2_max
            range2 = min(cache_size, max_margin)
            outbucket2 = [self.b2_cache.pop() for i in range(range2)]
        else:
            outbucket2 = []
        if len(self.b3_cache) > self.b3_max:
            cache_size = len(self.b3_cache)
            max_margin = self.b3_max
            range3 = min(cache_size, max_margin)
            outbucket3 = [self.b3_cache.pop() for i in range(range3)]
        else:
            outbucket3 = []

        # if len(outbucket0)==0:
        #     print(text)
        return outbucket0, outbucket1, outbucket2, outbucket3

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.size(), self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            # special_tokens_mask = special_tokens_mask.bool()

        # print(special_tokens_mask.size())
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.size(), 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.char2id.keys()), labels.size(), dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def pack_tensors(self, tokens):
        array = self.encoder(tokens)
        array = torch.nn.utils.rnn.pad_sequence(
            array, batch_first=True, padding_value=self.pad
        )
        # lengths = (array!=self.pad).sum(dim=-1)
        # Bert tokenization
        special_token_mask = [
            list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff))
            for stuff in array.tolist()
        ]
        masked_array, masked_labels = self.mask_tokens(array, special_token_mask)
        return masked_array, masked_labels  # , lengths

    def process(self, text):
        arrays = []
        # lengths = []
        targets = []
        # print(f"len text {len(text)}")
        # print(f"process_text {len(self.process_text(text))}")
        for tokens in self.process_text(text):
            if len(tokens) > 0:
                print(len(tokens))
                array, target = self.pack_tensors(tokens)
                arrays.append(array)
                targets.append(target)
            else:
                print("#########")
                # print(len(tokens))
                # print(text)
        # print(f"len array[0] {len(array[0])}")
        # print(f"len array[1] {len(array[1])}")
        # print(f"len array[2] {len(array[2])}")
        # print(f"len array[3] {len(array[3])}")
        # print(len(arrays[0]))
        return arrays


if __name__ == "__main__":

    text_encoder = Encoder()
