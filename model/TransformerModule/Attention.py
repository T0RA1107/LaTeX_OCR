import torch
import torch.nn as nn
import torch.nn.functional as F

class ScalarDotProductAttention(nn.Module):
    def __init__(
        self,
        n_head
    ):
        super().__init__()
        self.n_head = n_head

    def forward(
        self,
        Q,
        K,
        V,
        mask = None
    ):
        l_query, n_batch, n_head, d_query = Q.shape
        l_key, _, _, d_key = K.shape
        l_value, _, _, d_value = V.shape

        assert n_head == self.n_head, f"n_head: {n_head}, self.n_head: {self.n_head}"
        assert d_query == d_key, f"d_query: {d_query}, d_key: {d_key}"
        assert l_key == l_value, f"l_key: {l_key}, l_value: {l_value}"

        Q = Q.reshape(l_query, 1, n_batch, n_head, d_query, 1)
        K = K.reshape(1, l_key, n_batch, n_head, d_key, 1)
        V = V.reshape(1, l_value, n_batch, n_head, d_value)

        x = torch.sum(Q * K, dim=4)
        x = x / torch.sqrt(torch.tensor(d_key))

        if mask is not None:
            assert mask.shape == (l_query, l_key)
            x = x + mask[:, :, None, None, None]

        x = F.softmax(x, dim=1)  # (l_query, l_key, n_batch, n_head, 1)

        x = torch.sum(x * V, dim=1)  # (l_query, n_batch, n_head, d_value)
        return x

class LinearAttention(nn.Module):
    def __init__(
        self,
        n_head,
        eps=1e-6
    ):
        super().__init__()
        self.n_head = n_head
        self.eps = eps

    def forward(
        self,
        Q,
        K,
        V,
        mask = None
    ):
        assert mask is None, "LinearAttention does not support mask"
        l_query, n_batch, n_head, d_query = Q.shape
        l_key, _, _, d_key = K.shape
        l_value, _, _, d_value = V.shape

        assert n_head == self.n_head, f"n_head: {n_head}, self.n_head: {self.n_head}"
        assert d_query == d_key, f"d_query: {d_query}, d_key: {d_key}"
        assert l_key == l_value, f"l_key: {l_key}, l_value: {l_value}"

        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        Q = Q.reshape(l_query, n_batch, n_head, d_query, 1)
        K = K.reshape(1, l_key, n_batch, n_head, d_key, 1)
        V = V.reshape(1, l_value, n_batch, n_head, 1, d_value)

        A = torch.sum(K * V, dim=1)  # (1, n_batch, n_head, d_key, d_value)
        A = torch.sum(Q * A, dim=3)  # (l_query, n_batch, n_head, d_value)
        B = torch.sum(K, dim=1)  # (1, n_batch, n_head, d_key, 1)
        B = torch.sum(Q * B, dim=3)  # (l_query, n_batch, n_head, 1)

        x = A / (B + self.eps)
        return x

class CausalLinearAttention(nn.Module):
    def __init__(
        self,
        n_head,
        eps=1e-6
    ):
        super().__init__()
        self.n_head = n_head
        self.eps = eps

    def forward(
        self,
        Q,
        K,
        V,
        mask = None
    ):
        assert mask is None

        l_query, n_batch, n_head, d_query = Q.shape
        l_key, _, _, d_key = K.shape
        l_value, _, _, d_value = V.shape

        assert n_head == self.n_head, f"n_head: {n_head}, self.n_head: {self.n_head}"
        assert d_query == d_key, f"d_query: {d_query}, d_key: {d_key}"
        assert l_key == l_value, f"l_key: {l_key}, l_value: {l_value}"

        assert l_query == l_key, f"l_query: {l_query}, l_key: {l_key}\nCausal Linear Attention does not support non-square Attention map"

        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        Q = Q.reshape(l_query, n_batch, n_head, d_query, 1)
        K = K.reshape(l_key, n_batch, n_head, d_key, 1)
        V = V.reshape(l_value, n_batch, n_head, 1, d_value)

        A = torch.cumsum(K * V, dim=1)  # (l_key, n_batch, n_head, d_key, d_value)
        A = torch.sum(Q * A, dim=3)  # (l_query, n_batch, n_head, d_value)

        B = torch.cumsum(K, dim=1)  # (l_key, n_batch, n_head, d_key, 1)
        B = torch.sum(Q * B, dim=3)  # (l_query, n_batch, n_head, 1)

        x = A / (B + self.eps)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        d_query,
        d_key,
        d_value,
        n_head,
        attention_type="ScalarDotProductAttention"
    ):
        super().__init__()
        assert(d_query == d_key)
        self.d_model = d_model
        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head

        if attention_type == "ScalarDotProductAttention":
            self.attention = ScalarDotProductAttention(n_head)
        elif attention_type == "LinearAttention":
            self.attention = LinearAttention(n_head)
        elif attention_type == "CausalLinearAttention":
            self.attention = CausalLinearAttention(n_head)

        # for make Q, K, V into H heads
        self.project_query = nn.Linear(
            in_features=d_model,
            out_features=n_head * d_query
        )
        self.project_key = nn.Linear(
            in_features=d_model,
            out_features=n_head * d_key
        )
        self.project_value = nn.Linear(
            in_features=d_model,
            out_features=n_head * d_value
        )
        # for aggregate H heads
        self.aggregate =nn.Linear(
            in_features=n_head * d_value,
            out_features=d_model
        )

    def forward(
        self,
        Q,
        K,
        V,
        mask = None
    ):
        l_query, n_batch, d_model = Q.shape
        l_key, _, _ = K.shape
        l_value, _, _ = V.shape
        assert(l_key == l_value)
        assert(d_model == self.d_model)

        Q = Q.reshape(l_query * n_batch, d_model)
        K = K.reshape(l_key * n_batch, d_model)
        V = V.reshape(l_value * n_batch, d_model)

        Q = self.project_query(Q)
        K = self.project_key(K)
        V = self.project_value(V)

        Q = Q.reshape(l_query, n_batch, self.n_head, self.d_query)
        K = K.reshape(l_key, n_batch, self.n_head, self.d_key)
        V = V.reshape(l_value, n_batch, self.n_head, self.d_value)

        x = self.attention(Q, K, V, mask=mask)
        x = x.reshape(l_query * n_batch, self.n_head * self.d_value)
        x = self.aggregate(x)
        x = x.reshape(l_query, n_batch, self.d_model)
        return x
