
from collections import Counter
from itertools import pairwise

device = 'cuda'

device_type = 'cuda' if 'cuda' in device else 'cpu'

with open ("input.txt", "r", encoding="utf-8") as f:
    content = f.read(100)

class tokenization:
    def encoder(data):
        vocab = {}
        ID = []
        min_frequency = 2

        # Assign characters to a value
        for char in data:
            vocab.setdefault(char, len(vocab))
            ID.append(vocab[char])
        # Allow up to 100 learned tokens in addition to the starting characters.
        target_vocab_size = len(vocab) + 100

        # Each pass learns one new token from the current ID sequence.
        merge_number = 0
        while len(vocab) < target_vocab_size:
            # Recount pairs because the previous merge changed the sequence.
            pairs = Counter(pairwise(ID))

            # A sequence with fewer than two tokens has no pair to merge.
            if not pairs:
                break

            # Check the pair's total count before replacing any occurrences.
            chosen_pair = max(pairs, key=pairs.get)
            pair_frequency = pairs[chosen_pair]
            if pair_frequency < min_frequency:
                break

            # Merging pairs
            output = []
            l, r = 0, 1
            while l < len(ID):
                if r < len(ID):
                    cur = (ID[l], ID[r])
                    if cur == chosen_pair:
                        vocab.setdefault(cur, len(vocab))
                        output.append(vocab[cur])
                        l += 2
                        r += 2
                    else:
                        output.append(ID[l])
                        l += 1
                        r += 1
                else:
                    output.append(ID[l])
                    l += 1
                    r += 1

            # The next pass will count pairs in this newly merged sequence.
            ID = output
            merge_number += 1

        return vocab, ID

    vocab, ID = encoder(content)

    def decoder(vocab, ID):
        text_by_id = {}

        # Tokens were added in creation order, so each pair's children are ready.
        for piece, token_id in vocab.items():
            if isinstance(piece, str):
                text_by_id[token_id] = piece
            else:
                left_id, right_id = piece
                text_by_id[token_id] = text_by_id[left_id] + text_by_id[right_id]

        # Look up each token's full text directly, then join the pieces.
        return "".join(text_by_id[token_id] for token_id in ID)

    decoded_text = decoder(vocab, ID)