import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CNN(nn.Module):
    """
    Convolutional Neural Network for character-level language modeling.

    Args:
        vocab_size (int): Number of unique characters in the vocabulary.
        embedding_dim (int): Dimensionality of the character embeddings (default: 24).
        hidden_dim (int): Number of neurons in the hidden layers (default: 128).
    """
    
    def __init__(self, vocab_size, embedding_dim=24, hidden_dim=128):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2)
        
        # Linear output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights for convolutional layers
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        
    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        x = self.embedding(x)  # Embedding
        x = F.relu(self.conv1(x.permute(0, 2, 1)))  # Conv1
        x = F.relu(self.conv2(x))  # Conv2
        x = F.relu(self.conv3(x))  # Conv3
        x = self.fc(x.mean(dim=2))  # Global average pooling over time dimension
        return x
    
    def build_dataset(self, words, block_size=8):
        """
        Builds the dataset for training.

        Args:
            words (list): List of words (strings).
            block_size (int): Size of the context window (default: 8).
        """
        X, Y = [], []
        stoi = {s: i+1 for i, s in enumerate(sorted(list(set(''.join(words)))))}
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
    
    def train(self, max_steps=200000, batch_size=32):
        """
        Trains the CNN model.

        Args:
            max_steps (int): Maximum number of training steps (default: 200000).
            batch_size (int): Size of each minibatch (default: 32).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        loss_log = []
        for i in range(max_steps):
            ix = torch.randint(0, self.X.shape[0], (batch_size,))
            Xb, Yb = self.X[ix], self.Y[ix]
            
            optimizer.zero_grad()
            logits = self(Xb)
            loss = criterion(logits, Yb)
            loss.backward()
            optimizer.step()
            
            if i % 10000 == 0:
                print(f'{i}/{max_steps}: {loss.item():.4f}')
            loss_log.append(loss.item())
    
    @torch.no_grad()
    def evaluate(self, split='test'):
        """
        Evaluates the CNN model on a dataset split.

        Args:
            split (str): Dataset split to evaluate ('train', 'val', or 'test', default: 'test').
        """
        if split == 'train':
            X, Y = self.X[:int(0.8 * len(self.X))], self.Y[:int(0.8 * len(self.Y))]
        elif split == 'val':
            X, Y = self.X[int(0.8 * len(self.X)):int(0.9 * len(self.X))], \
                   self.Y[int(0.8 * len(self.Y)):int(0.9 * len(self.Y))]
        elif split == 'test':
            X, Y = self.X[int(0.9 * len(self.X)):], self.Y[int(0.9 * len(self.Y)):]
        
        logits = self(X)
        loss = F.cross_entropy(logits, Y)
        print(f'{split.capitalize()} loss: {loss.item():.4f}')
    
    @torch.no_grad()
    def sample(self, stoi, num_samples=20, block_size=8):
        """
        Samples from the trained CNN model.

        Args:
            stoi (dict): Dictionary mapping characters to their indices.
            num_samples (int): Number of samples to generate (default: 20).
            block_size (int): Size of the context window (default: 8).
        """
        for _ in range(num_samples):
            out = []
            context = [0] * block_size
            while True:
                x = torch.tensor([context])
                logits = self(x)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            print(''.join([list(stoi.keys())[list(stoi.values()).index(i)] for i in out]))