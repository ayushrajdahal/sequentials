import torch
import torch.nn.functional as F
import random


class Ngram:
    def __init__(self, names, context_length=3, embed_dim=10, mid_layer_dim=100, learning_rate=0.1, seed=42):
        """
        Initializes the MLP model with the given parameters.

        Parameters
        ----------
        names : list of str
            List of names to be used for training the model.
        context_length : int, optional
            Length of the context (default is 3).
        embed_dim : int, optional
            Dimension of the embedding (default is 10).
        mid_layer_dim : int, optional
            Dimension of the middle layer (default is 100).
        learning_rate : float, optional
            Learning rate for training (default is 0.1).
        """
        self.names = names
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.mid_layer_dim = mid_layer_dim
        self.learning_rate = learning_rate
        
        self.idx_to_str = {i: s for i, s in enumerate(["."] + sorted(list(set(''.join(names)))))}
        self.str_to_idx = {s: i for i, s in self.idx_to_str.items()}
        
        self.generator = torch.Generator().manual_seed(seed)
        random.seed(420)

        self.init_params()

    def init_params(self):
        """Initializes the parameters of the model."""

        self.C = torch.randn((27, self.embed_dim), generator=self.generator)
        self.W1 = torch.randn((self.embed_dim * self.context_length, self.mid_layer_dim), generator=self.generator)
        self.b1 = torch.randn(self.mid_layer_dim, generator=self.generator)
        self.W2 = torch.randn((self.mid_layer_dim, 27), generator=self.generator)
        self.b2 = torch.randn(27, generator=self.generator)

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]

        for param in self.params:
            param.requires_grad = True


    def create_dataset(self, input_names):
        """
        Creates a dataset of contexts and targets from input names.

        Parameters
        ----------
        input_names : list of str
            List of names to generate the dataset from.

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing the contexts and targets tensors.
        """
        db_contexts = []
        db_targets = []
        for name in input_names:
            context = [0] * self.context_length
            for char in name + '.':
                char_idx = self.str_to_idx[char]
                db_contexts.append(context)
                db_targets.append(char_idx)
                context = context[1:] + [char_idx]

        db_contexts = torch.tensor(db_contexts)
        db_targets = torch.tensor(db_targets)

        return db_contexts, db_targets


    def split_dataset(self):
        """Splits the names into training, validation, and test datasets."""
        random.shuffle(self.names)
        n1 = int(0.7 * len(self.names))
        n2 = int(0.8 * len(self.names))

        X_train, Y_train = self.create_dataset(self.names[:n1])
        X_test, Y_test = self.create_dataset(self.names[n1:n2])
        X_val, Y_val = self.create_dataset(self.names[n2:])

        self.dataset = {
            "train": (X_train, Y_train),
            "test": (X_test, Y_test),
            "val": (X_val, Y_val)
        }


    def train(self, steps=100000, batch_size=32):
        """
        Trains the MLP model.

        Parameters
        ----------
        steps : int, optional
            Number of training steps (default is 100000).
        batch_size : int, optional
            Size of the mini-batch for training (default is 32).
        """
        self.split_dataset()
        X, Y = self.dataset["train"]

        step_log = []
        loss_log = []

        for i in range(steps):
            ix = torch.randint(0, X.shape[0], (batch_size,))
            emb = self.C[X[ix]].view(-1, self.context_length * self.embed_dim)
            h = torch.tanh(emb @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, Y[ix])

            for param in self.params:
                param.grad = None
            loss.backward()

            learning_rate = self.learning_rate if i < 100000 else 0.01
            for param in self.params:
                param.data += -learning_rate * param.grad

            step_log.append(i)
            loss_log.append(loss.log10().item())

        print(f'Average loss over the last 100 steps: {sum(loss_log[-100:]) / 100:.4f}')


    def test(self):
        """Calculates and prints the test loss."""
        X, Y = self.dataset["test"]
        emb = self.C[X].view(-1, self.context_length * self.embed_dim)
        h = torch.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, Y)
        print(f'Test loss: {loss.item():.4f}')


    def sample(self, num_samples=10):
        """
        Generates samples from the trained MLP model.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate (default is 10).
        """
        for _ in range(num_samples):
            context = [0] * self.context_length
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                counts = logits.exp()
                prob = counts / counts.sum(1, keepdim=True)
                ix = torch.multinomial(prob, num_samples=1, generator=self.generator).item()
                context = context[1:] + [ix]

                if ix == 0:
                    print()
                    break
                print(f'{self.idx_to_str[ix]}', end='')
