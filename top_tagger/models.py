import torch,math
import torch.nn as nn
from torch.nn import functional as F


# import preprocess as pre

# path = "/home/tuhin/Python codes and all/"

# X,P4,mask,label = pre.prepare_jet_data(path = path,
#                             max_particles = 128,
#                             max_sample_per_class= 8000,
#                             seed = 0)

# jet_dataset = pre.JetDataSet(X_tensor= X,
#                             P4_tensor = P4,
#                             mask_tensor = mask,
#                             label_tensor = label)

# loader = pre.create_loader(jet_dataset=jet_dataset,
#                            train_frac=0.7,
#                            test_frac=0.15)
def scaled_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_bias=None,
    mask=None
):
    """
    Compute scaled dot-product attention.

    Args:
        Q (torch.Tensor): Query tensor of shape (B, H, N_q, d_k).
        K (torch.Tensor): Key tensor of shape (B, H, N_k, d_k).
        V (torch.Tensor): Value tensor of shape (B, H, N_k, d_k).
        attn_bias (torch.Tensor, optional): Additive bias applied to the
            attention scores before softmax, shape (B, H, N_q, N_k).
            Used to inject pairwise interaction features (e.g. Particle
            Transformer's U matrix). Default: None.
        mask (torch.Tensor, optional): Boolean mask of shape (B, N_k)
            indicating valid (True) vs padded (False) key positions.
            Padded positions are set to -inf before softmax. Default: None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - out: Attention output, shape (B, H, N_q, d_k).
            - attention_matrix: Softmax attention weights, shape
              (B, H, N_q, N_k).
    """
    dim_k = K.shape[-1]
    score_matrix = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1)) / math.sqrt(dim_k)

    if attn_bias is not None:
        score_matrix = score_matrix + attn_bias
    if mask is not None:
        score_matrix = score_matrix.masked_fill(~mask[:, None, None, :], float("-inf"))
    attention_matrix = F.softmax(score_matrix, dim=-1)
    attention_matrix = torch.nan_to_num(attention_matrix)
    out = attention_matrix @ V

    return out, attention_matrix


def pairwise_features(P4, eps=1e-8):
    """
    Compute pairwise interaction features between particles from their
    4-momenta, used as the bias input (U) to Particle Attention Blocks.

    Args:
        P4 (torch.Tensor): Particle 4-momenta of shape (B, N, 4), with
            components (px, py, pz, E) along the last dimension.
        eps (float, optional): Small constant for numerical stability in
            the rapidity calculation. Default: 1e-8.

    Returns:
        torch.Tensor: Pairwise interaction features of shape (B, N, N, 4),
            containing log(delta), log(kt), log(z), and log(m2) for every
            particle pair (i, j):
                - delta: angular distance in (rapidity, phi) space.
                - kt: relative transverse momentum scale.
                - z: momentum fraction.
                - m2: invariant mass squared of the pair.
    """
    px, py, pz, E = P4[..., 0], P4[..., 1], P4[..., 2], P4[..., 3]
    pt  = torch.sqrt(px**2 + py**2).clamp(min=1e-6)
    phi = torch.atan2(py, px)
    y   = 0.5 * torch.log(((E + pz).clamp(min=eps)) / ((E - pz).clamp(min=eps)))     # rapidity

    dy   = y[:, :, None] - y[:, None, :]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))                              # wrap to (-pi, pi], MPS-safe
    delta = torch.sqrt(dy**2 + dphi**2).clamp(min=1e-6)

    pt_min = torch.minimum(pt[:, :, None], pt[:, None, :])
    kt = (pt_min * delta).clamp(min=1e-6)
    z  = (pt_min / (pt[:, :, None] + pt[:, None, :] + eps)).clamp(min=1e-6, max=1.0)
    sum_E, sum_px = E[:, :, None] + E[:, None, :], px[:, :, None] + px[:, None, :]
    sum_py, sum_pz = py[:, :, None] + py[:, None, :], pz[:, :, None] + pz[:, None, :]
    m2 = (sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2).clamp(min=1e-6)

    U = torch.stack([torch.log(delta), torch.log(kt), torch.log(z), torch.log(m2)], dim=-1)

    return U


class MultiHeadAttentionBlock(nn.Module):
    """
    Particle Attention Block (Figure 3b).

    Applies pre-norm multi-head self-attention with an optional additive
    interaction bias (U), followed by a pre-norm MLP, each wrapped in a
    residual connection:

        x -> LN -> MHA(+attn_bias) -> LN -> (+x) -> MLP -> (+prev)

    Args:
        emb_dim (int): Embedding dimension D of the input tokens.
        head (int): Number of attention heads.
        ratio (int, optional): Expansion ratio for the hidden layer of the
            MLP (hidden dim = ratio * emb_dim). Default: 2.
    """

    def __init__(self, emb_dim, head, ratio=2):
        """
        Initialize the Particle Attention Block's layers.

        Args:
            emb_dim (int): Embedding dimension D.
            head (int): Number of attention heads. Must evenly divide emb_dim.
            ratio (int, optional): MLP hidden-layer expansion ratio. Default: 2.
        """
        super().__init__()
        assert emb_dim % head == 0
        self.dim_head, self.dim_k = head, emb_dim // head
        self.QKV = nn.Linear(emb_dim, 3 * emb_dim)  # (D,3D)
        self.ln1, self.ln2 = nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ratio * emb_dim),
            nn.GELU(),
            nn.LayerNorm(ratio * emb_dim),
            nn.Linear(ratio * emb_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor, mask=None, attn_bias=None):
        """
        Run one Particle Attention Block forward pass.

        Args:
            x (torch.Tensor): Input particle embeddings, shape (B, N, D).
            mask (torch.Tensor, optional): Boolean padding mask of shape
                (B, N), True for valid particles. Default: None.
            attn_bias (torch.Tensor, optional): Pairwise interaction bias
                U of shape (B, H, N, N), added to attention scores before
                softmax. Default: None.

        Returns:
            torch.Tensor: Updated particle embeddings, shape (B, N, D).
        """
        B, N, D = x.shape
        normed_x = self.ln1(x)

        QKV = self.QKV(normed_x)  # (B,N,3D)
        QKV = QKV.reshape(B, N, 3, self.dim_head, self.dim_k).permute(2, 0, 3, 1, 4)  # (3,B,H,N,dk)
        Q, K, V = QKV[0], QKV[1], QKV[2]  # Q,K,V : (B,H,N,dk)

        out, attention = scaled_attention(Q=Q, K=K, V=V, mask=mask, attn_bias=attn_bias)  # out : (B,H,N,dk)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B,N,D)

        out = self.ln2(out)
        out = out + x

        new_out = self.mlp(out)

        return out + new_out


class ClassAttentionBlock(nn.Module):
    """
    Class Attention Block (Figure 3c).

    A learnable class token attends to itself concatenated with the
    particle embeddings via cross-attention, where the query comes from
    the raw (un-normalized) class token and the keys/values come from the
    normalized concatenation. Followed by a pre-norm MLP, each wrapped in
    a residual connection.

    Args:
        emb_dim (int): Embedding dimension D.
        head (int): Number of attention heads.
        ratio (int, optional): Expansion ratio for the hidden layer of the
            MLP (hidden dim = ratio * emb_dim). Default: 2.
    """

    def __init__(self, emb_dim, head, ratio=2):
        """
        Initialize the Class Attention Block's layers and learnable class token.

        Args:
            emb_dim (int): Embedding dimension D.
            head (int): Number of attention heads. Must evenly divide emb_dim.
            ratio (int, optional): MLP hidden-layer expansion ratio. Default: 2.
        """
        super().__init__()
        assert emb_dim % head == 0
        self.dim_head, self.dim_k = head, emb_dim // head
        self.x_class = nn.Parameter(torch.randn([1, 1, emb_dim]))

        self.ln1, self.ln2 = nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim)
        self.Q, self.KV = nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, 2 * emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ratio * emb_dim),
            nn.GELU(),
            nn.LayerNorm(ratio * emb_dim),
            nn.Linear(ratio * emb_dim, emb_dim)
        )

    def forward(self, x, mask, modified_x_class=None):
        """
        Run one Class Attention Block forward pass.

        Args:
            x (torch.Tensor): Particle embeddings, shape (B, N, D).
            mask (torch.Tensor): Boolean padding mask of shape (B, N),
                True for valid particles.
            modified_x_class (torch.Tensor, optional): Class token carried
                over from a previous Class Attention Block, shape
                (B, 1, D). If None, uses this block's own learnable
                x_class parameter (broadcast to batch size). Default: None.

        Returns:
            torch.Tensor: Updated class token, shape (B, 1, D).
        """
        B, N, D = x.shape
        x_class = self.x_class.expand([B, -1, -1])
        if modified_x_class is not None:
            x_class = modified_x_class
        x_cat = torch.concat((x_class, x), dim=-2)
        normed_x_cat = self.ln1(x_cat)

        Q = self.Q(x_class)
        Q = Q.reshape(B, 1, self.dim_head, self.dim_k).permute(0, 2, 1, 3)  # (B,H,1,dk)
        KV = self.KV(normed_x_cat)  # (B,N+1,2D)
        KV = KV.reshape(B, N + 1, 2, self.dim_head, self.dim_k).permute(2, 0, 3, 1, 4)  # (2,B,H,N+1,dk)
        K, V = KV[0], KV[1]  # K,V : (B,H,N+1,dk)

        class_mask = torch.ones(B, 1, dtype=mask.dtype,)
        class_attn_mask = torch.cat([class_mask, mask], dim=1)

        out, attention = scaled_attention(Q=Q, K=K, V=V, mask=class_attn_mask)  # out : (B,H,N,dk)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, D)  # (B,N,D)

        out = self.ln2(out)
        out = out + x_class

        new_out = self.mlp(out)
        return out + new_out


class ParticleTransformer(nn.Module):
    """
    Particle Transformer (Figure 3a).

    Projects raw particle features into an embedding space, refines them
    through a stack of Particle Attention Blocks (optionally biased by
    pairwise interaction features derived from 4-momenta), then pools
    information into a class token through two Class Attention Blocks,
    and produces final classification logits via an MLP head.

    Args:
        in_feat (int): Number of raw input features per particle.
        out_feat (int): Number of output classes.
        emb_dim (int): Embedding dimension D used throughout the network.
        head (int): Number of attention heads.
        depth (int, optional): Number of Particle Attention Blocks. Default: 3.
        use_bias (bool, optional): Whether to compute and use pairwise
            interaction features (U) as an attention bias. Default: True.
    """

    def __init__(self, in_feat, out_feat, emb_dim, head, depth=3, use_bias=True):
        """
        Initialize the Particle Transformer's submodules.

        Args:
            in_feat (int): Number of raw input features per particle.
            out_feat (int): Number of output classes.
            emb_dim (int): Embedding dimension D.
            head (int): Number of attention heads.
            depth (int, optional): Number of Particle Attention Blocks. Default: 3.
            use_bias (bool, optional): Whether to build the interaction-feature
                MLP used to embed pairwise features into an attention bias.
                Default: True.
        """
        super().__init__()

        self.use_bias = use_bias

        self.input_projection = nn.Linear(in_feat, emb_dim)
        self.particle_attn_section = nn.ModuleList([MultiHeadAttentionBlock(emb_dim, head) for _ in range(depth)])
        self.class_attn_1 = ClassAttentionBlock(emb_dim, head)
        self.class_attn_2 = ClassAttentionBlock(emb_dim, head)
        self.MLP_head = nn.Linear(emb_dim, out_feat)

        if self.use_bias:
            self.interaction_mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.GELU(),
                nn.Linear(16, head)
            )

    def forward(self, x, mask, P4=None):
        """
        Run the full Particle Transformer forward pass.

        Args:
            x (torch.Tensor): Raw particle features, shape (B, N, in_feat).
            mask (torch.Tensor): Boolean padding mask of shape (B, N),
                True for valid particles.
            P4 (torch.Tensor, optional): Particle 4-momenta of shape
                (B, N, 4), used to compute the pairwise interaction bias
                when use_bias=True. Default: None.

        Returns:
            torch.Tensor: Classification logits, shape (B, out_feat).
        """
        attn_bias = None

        if P4 is not None and self.use_bias:
            U = pairwise_features(P4=P4)  # (B,N,N,4)
            attn_bias = self.interaction_mlp(U) # (B,N,N,H)
            attn_bias = attn_bias.permute(0, 3, 1, 2)

        emb_x = self.input_projection(x)

        for block in self.particle_attn_section:
            emb_x = block(emb_x, mask, attn_bias)

        x_class_mod = self.class_attn_1(emb_x, mask)

        output = self.class_attn_2(emb_x, mask, modified_x_class=x_class_mod)
        output = output.squeeze(dim=1)

        logits = self.MLP_head(output)

        return logits