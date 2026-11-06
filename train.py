
device = 'cuda'

device_type = 'cuda' if 'cuda' in device else 'cpu'

with open ("input.txt", "r", encoding="utf-8") as f:
    content = f.read()

pr = "abcab"

class tokenization:
    vocab = {}
    ID = []

    # assign characters to a value
    for char in content[:1000]:
        vocab.setdefault(char, len(vocab))
        ID.append(vocab[char])

    # pairing values
    pairs = {}
    i = 0
    while i + 1 < len(ID):
        pair = ID[i], ID[i + 1]
        pairs[pair] = pairs.get(pair, 0) + 1
        i += 1

    # merging pairs
    output = []
    l, r = 0, 1
    chosen_pair = max(pairs, key=pairs.get, default=None)
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

    print(f'ID: {ID} \nvocab: {vocab} \npairs: {pairs} \noutput: {output}')