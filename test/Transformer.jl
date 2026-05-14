# A minimal GPT-style Transformer implementation in pure Julia
# Inpired from https://github.com/karpathy/nanogpt

using Random
using LinearAlgebra

# Helper functions
function gelu(x)
    return 0.5 * x .* (1 .+ tanh.(sqrt(2 / π) .* (x .+ 0.044715 .* x .^ 3)))
end

# LayerNorm
struct LayerNorm{V}
    γ::V
    β::V
    ϵ::Float64
end

function LayerNorm(dim::Int; ϵ = 1e-5)
    # We could use `ones(dim)` and thend
    # do `γ'` but then we'll need to implement
    # `adjoint` for `VectNode`
    γ = ones(1, dim)
    β = zeros(1, dim)
    return LayerNorm(γ, β, ϵ)
end

function (ln::LayerNorm)(x)
    d = size(x, 2)
    μ = sum(x, dims = 2) / d
    σ2 = sum((x .- μ) .^ 2, dims = 2) / d
    x̂ = (x .- μ) ./ sqrt.(σ2 .+ ln.ϵ)
    return ln.γ .* x̂ .+ ln.β
end

# Causal Self-Attention (single head)
struct CausalSelfAttention{M}
    wq::M
    wk::M
    wv::M
end

function CausalSelfAttention(d_emb::Int, d_head::Int)
    wq = randn(d_emb, d_head) * 0.02
    wk = randn(d_emb, d_head) * 0.02
    wv = randn(d_emb, d_head) * 0.02
    return CausalSelfAttention(wq, wk, wv)
end

function (attn::CausalSelfAttention)(x)
    # x: (seq, d_emb)
    q = x * attn.wq
    k = x * attn.wk
    v = x * attn.wv

    seq = size(x, 1)
    d_head = size(attn.wq, 2)

    attn_scores = (q * k') / sqrt(d_head)
    # Causal mask
    mask = [i < j ? -Inf : Inf for i in 1:seq, j in 1:seq]
    attn_scores = min.(attn_scores, mask)
    attn_weights = softmax(attn_scores, dims = 2)
    return attn_weights * v
end

# Multi-Head Attention
struct MultiHead{M}
    heads::Vector{CausalSelfAttention{M}}
    wo::M
end

function MultiHead(d_emb::Int, n_head::Int)
    head_dim = div(d_emb, n_head)
    heads = [CausalSelfAttention(d_emb, head_dim) for _ in 1:n_head]
    wo = randn(n_head * head_dim, d_emb) * 0.02
    return MultiHead(heads, wo)
end

function (mha::MultiHead)(x)
    outs = [head(x) for head in mha.heads]
    out = reduce(hcat, outs) # (seq, n_head*head_dim)
    return out * mha.wo
end

# MLP (Feedforward network)
struct MLP{M}
    c_fc::M
    c_proj::M
end

function MLP(d_emb::Int, d_hidden::Int)
    c_fc = randn(d_emb, d_hidden) * 0.02
    c_proj = randn(d_hidden, d_emb) * 0.02
    return MLP(c_fc, c_proj)
end

function (mlp::MLP)(x)
    return gelu(x * mlp.c_fc) * mlp.c_proj
end

# Transformer Block
struct Block{V,M}
    ln1::LayerNorm{V}
    attn::MultiHead
    ln2::LayerNorm{V}
    mlp::MLP{M}
end

function Block(d_emb::Int, n_head::Int, n_hidden::Int)
    ln1 = LayerNorm(d_emb)
    attn = MultiHead(d_emb, n_head)
    ln2 = LayerNorm(d_emb)
    mlp = MLP(d_emb, n_hidden)
    return Block(ln1, attn, ln2, mlp)
end

function (block::Block)(x)
    x = x .+ block.attn(block.ln1(x))
    x = x .+ block.mlp(block.ln2(x))
    return x
end

# The full Transformer
struct Transformer{V,M}
    wte::M  # token embedding
    wpe::M  # position embedding
    blocks::Vector{Block{V,M}}
    ln_f::LayerNorm{V}
    n_voc::Int
    d_emb::Int
end

function Transformer(;
    n_voc::Int,
    n_ctx::Int,
    n_layer::Int,
    n_head::Int,
    d_emb::Int,
    d_ff::Int,
)
    wte = randn(n_voc, d_emb) * 0.02
    wpe = randn(n_ctx, d_emb) * 0.02
    blocks = [Block(d_emb, n_head, d_ff) for _ in 1:n_layer]
    ln_f = LayerNorm(d_emb)
    return Transformer(wte, wpe, blocks, ln_f, n_voc, d_emb)
end

function (m::Transformer)(idx)
    x = m.wte[idx, :] .+ m.wpe[eachindex(idx), :]
    for block in m.blocks
        x = block(x)
    end
    x = m.ln_f(x)
    # logits: (seq, n_voc)
    logits = x * m.wte'
    return logits
end

# Softmax helper
function softmax(x; dims = 1)
    x_max = maximum(x, dims = dims)
    ex = exp.(x .- x_max)
    return ex ./ sum(ex, dims = dims)
end
