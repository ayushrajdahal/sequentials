import torch
import torch.nn.functional as F


class Bigram:
    def __init__(self, words):
        self.words = words
        self.generator = torch.Generator()
        self.stoi, self.itos = self.create_mappings(self.words)
        self.vocab_size = len(self.stoi.keys())
        self.xs = self.tokenize_names(self.words)
        self.W = torch.randn((self.vocab_size, self.vocab_size), generator=self.generator, requires_grad=True)


    def create_mappings(self, words):
        '''
        create character-to-index and index-to-character mappings

        '''
        chars = ['.']+sorted(list(set(''.join(words))))
        str_to_index = {s: i for i, s in enumerate(chars)}
        index_to_str = {i: s for s, i in str_to_index.items()}
        return str_to_index, index_to_str


    def tokenize_names(self, names):
        '''
        tokenize names, character-wise

        '''
        xs = []
        for name in names:
            xs += [self.stoi[char] for char in ['.'] + list(name) + ['.']]
        return xs


    def compute_loss(self, xs, ys):
        '''
        compute loss using negative log likelihood

        '''

        one_hot_encoded_x = F.one_hot(xs, num_classes=self.vocab_size).float()
        logits = one_hot_encoded_x @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (self.W ** 2).mean()
        return loss


    def train_model(self, learning_rate=10, epochs=500):
        '''
        train the model

        '''

        xs = torch.tensor(self.xs[:-1], dtype=torch.long)
        ys = torch.tensor(self.xs[1:], dtype=torch.long)

        for epoch in range(epochs):
            loss = self.compute_loss(self.W, xs, ys)
            if (epoch) % 100 == 0:
                print(f'Epoch {epoch}, loss={loss.item():.4f}')
            self.W.grad = None
            loss.backward()
            self.W.data -= learning_rate * self.W.grad


    def sample_model(self, sample_size=10):
        '''
        sample from the model
        
        '''

        samples = []

        for _ in range(sample_size):
            out = []
            index = 1
            while True:
                equiv_one_hot = F.one_hot(torch.tensor([index]), num_classes=self.vocab_size).float()
                logits = equiv_one_hot @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdim=True)
                index = torch.multinomial(p, num_samples=1, replacement=True, generator=self.generator).item()
                out.append(self.itos[index])
                if index == 0:
                    break
            samples.append(''.join(out))

        samples_copy = []

        for sample in samples:
            sample = sample.replace('.', '')
            if len(sample) != 0:
                samples_copy.append(sample)
        
        return samples