import torch
import torch.nn.functional as F

class Bigram:
    def __init__(self, words, seed=42):
        """
        Initializes the Bigram model with the given list of words.

        Parameters
        ----------
        words : list of str
            List of words to be used for training the bigram model.
        
        seed : int
            Seed value for ensuring reproducability.
        """
        self.words = words
        self.generator = torch.Generator().manual_seed(seed)
        self.stoi, self.itos = self.create_mappings(self.words)
        self.vocab_size = len(self.stoi.keys())
        self.xs = self.tokenize_names(self.words)
        self.W = torch.randn((self.vocab_size, self.vocab_size), generator=self.generator, requires_grad=True)

    def create_mappings(self, words):
        """
        Creates character-to-index and index-to-character mappings.

        Parameters
        ----------
        words : list of str
            List of words to create mappings from.

        Returns
        -------
        tuple of dict
            A tuple containing two dictionaries:
            - str_to_index : dict
                Mapping from character to index.
            - index_to_str : dict
                Mapping from index to character.
        """
        chars = sorted(list(set(''.join(words))))
        str_to_index = {s: i for i, s in enumerate(chars)}
        index_to_str = {i: s for s, i in str_to_index.items()}
        return str_to_index, index_to_str

    def tokenize_names(self, names):
        """
        Tokenizes names into character indices using the created mappings.

        Parameters
        ----------
        names : list of str
            List of names to tokenize.

        Returns
        -------
        list of int
            List of tokenized character indices.
        """
        xs = []
        for name in names:
            xs += [self.stoi[char] for char in ['.'] + list(name) + ['.']]
        return xs

    def compute_loss(self, xs, ys):
        """
        Computes the negative log likelihood loss for the bigram model.

        Parameters
        ----------
        xs : torch.Tensor
            Input tensor of shape (N,) where N is the sequence length.
        ys : torch.Tensor
            Target tensor of shape (N,) where N is the sequence length.

        Returns
        -------
        float
            Computed loss value.
        """
        one_hot_encoded_x = F.one_hot(xs, num_classes=self.vocab_size).float()
        logits = one_hot_encoded_x @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (self.W ** 2).mean()
        return loss.item()

    def train(self, learning_rate=10, epochs=500):
        """
        Trains the bigram model using stochastic gradient descent.

        Parameters
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        epochs : int
            Number of training epochs.
        """
        xs = torch.tensor(self.xs[:-1], dtype=torch.long)
        ys = torch.tensor(self.xs[1:], dtype=torch.long)

        for epoch in range(epochs):
            loss = self.compute_loss(xs, ys)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, loss={loss.item():.4f}')
            self.W.grad = None
            loss.backward()
            self.W.data -= learning_rate * self.W.grad

    def sample(self, sample_size=10):
        """
        Generates samples from the trained bigram model.

        Parameters
        ----------
        sample_size : int
            Number of samples to generate.

        Returns
        -------
        list of str
            List of generated samples.
        """
        samples = []

        for _ in range(sample_size):
            out = []
            index = 0  # Start with the index for '.'
            while True:
                equiv_one_hot = F.one_hot(torch.tensor([index]), num_classes=self.vocab_size).float()
                logits = equiv_one_hot @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdim=True)
                index = torch.multinomial(p, num_samples=1, replacement=True, generator=self.generator).item()
                if index == 0:  # Stop if we sample the index for '.'
                    break
                out.append(self.itos[index])
            samples.append(''.join(out))

        return samples
