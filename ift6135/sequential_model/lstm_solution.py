import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        vocabulary_size=40479,
        embedding_size=768,
        hidden_size=512,
        num_layers=1,
        learn_embeddings=False,
        _embedding_weight=None,
    ):

        super(LSTM, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_embeddings = learn_embeddings

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0, _weight=_embedding_weight
        )
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, vocabulary_size, bias=False),
        )

        # Tying classifier and embedding weights (similar to GPT-1)
        self.classifier[2].weight = self.embedding.weight

        # Freeze the embedding weights, depending on learn_embeddings
        self.embedding.requires_grad_(learn_embeddings)

    def forward(self, inputs, hidden_states):
        """LSTM.

        This is a Long Short-Term Memory network for language modeling. This
        module returns for each position in the sequence the log-probabilities
        of the next token. See Lecture 05, slides 42-60.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states (`tuple` of size 2)
            The (initial) hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch. For example, `log_probas[0, 3, 6]`
            corresponds to log p(x_{5} = token_{7} | x_{0:4}) (x_{5} for the word
            after x_{4} at index 3, and token_{7} for index 6) for the 1st sequence
            of the batch (index 0).

        hidden_states (`tuple` of size 2)
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # print('input_dimension: {}'.format(inputs.shape))
        embeds = self.embedding(inputs)
        # print('embeds_dimension: {}'.format(embeds.shape))
        # print('hidden_states : {}'.format(hidden_states.shape))
        lstm_out, (h, c) = self.lstm(embeds, hidden_states)
        # print('lstm_out_dimension: {}'.format(lstm_out.shape))
        # print('h_c_dimension: {}'.format(h.shape))
        # print('h_c_dimension: {}'.format(c.shape))
        tag_space = self.classifier(lstm_out)
        print('tag_space_dimension: {}'.format(tag_space.shape))
        tag_scores = F.log_softmax(tag_space, dim=2)

        print('tag_score_dimension: {}'.format(tag_scores.shape))
        # print(tag_scores.sum(dim=2))

        return tag_scores, (h, c)

    def loss(self, log_probas, targets, mask):
        """Loss function.

        This function computes the loss (negative log-likelihood).

        Parameters
        ----------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch.

        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing the target next tokens for all positions in
            each sequence of the batch.

        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing values in {0, 1} only, where the value is 0
            for positions corresponding to padding in the sequence, and 1
            otherwise.

        Returns
        -------
        loss (`torch.FloatTensor` scalar)
            The scalar loss, corresponding to the (mean) negative log-likelihood.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        # loss = torch.zeros(mask.shape[0])
        # for i in range(mask.shape[0]):
        #     t = 0
        #     for j in range(mask.shape[1]):
        #
        #         if mask[i][j] == 1:
        #             t += 1
        #             loss[i] += log_probas[i, j, targets[i, j]]
        #     loss[i] = loss[i]/t
        #
        # out = -loss.mean()
        nll = torch.nn.NLLLoss(reduction='none')
        log_probas = log_probas.transpose(1, 2)
        output = nll(log_probas, targets)
        output = output * mask
        loss = (torch.sum(output, dim=1) / torch.sum(mask, dim=1)).mean()
        return loss

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers, batch_size, self.hidden_size)

        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        c_0 = torch.zeros(shape, dtype=torch.float, device=device)

        return (h_0, c_0)

    @classmethod
    def load_embeddings_from(
        cls, filename, hidden_size=512, num_layers=1, learn_embeddings=False
    ):
        # Load the token embeddings from filename
        with open(filename, "rb") as f:
            embeddings = np.load(f)
            weight = torch.from_numpy(embeddings["tokens"])

        vocabulary_size, embedding_size = weight.shape
        return cls(
            vocabulary_size,
            embedding_size,
            hidden_size,
            num_layers,
            learn_embeddings,
            _embedding_weight=weight,
        )
