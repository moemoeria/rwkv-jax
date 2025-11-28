import os
import math
import time
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax, random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================


class RWKVConfig:
    def __init__(self):
        self.vocab_size = 65536
        self.n_embd = 512  # Channel dimensions
        self.n_layer = 4  # Number of layers (small for demo)
        self.ctx_len = 128  # Context length
        self.batch_size = 16  # Batch size
        self.head_size = 64  # v7 head size
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

        # LoRA dimensions for v7 params
        self.n_head = self.n_embd // self.head_size
        self.decay_lora = 64
        self.aaa_lora = 64
        self.mv_lora = 32
        self.gate_lora = 128

        # Training
        self.lr_init = 4e-3
        self.lr_final = 1e-5
        self.total_steps = 500
        self.seed = 42


config = RWKVConfig()

# ==============================================================================
# 2. Model Utilities
# ==============================================================================


def rwkv_orthogonal_init(scale=1.0):
    """Orthogonal initialization adapted for RWKV."""

    def init(key, shape, dtype=jnp.float32):
        if len(shape) == 2:
            gain = scale
            if shape[0] > shape[1]:
                gain = scale * math.sqrt(shape[0] / shape[1])
            return nn.initializers.orthogonal(scale=gain)(key, shape, dtype)
        elif len(shape) == 3:
            # For shapes like (C, H, W) or specific RWKV params, init per slice
            keys = random.split(key, shape[0])
            gain = scale
            if shape[1] > shape[2]:
                gain = scale * math.sqrt(shape[1] / shape[2])
            return jnp.stack(
                [
                    nn.initializers.orthogonal(scale=gain)(k, shape[1:], dtype)
                    for k in keys
                ]
            )
        return nn.initializers.uniform(scale)(key, shape, dtype)

    return init


# ==============================================================================
# 3. Model Definition (RWKV v7)
# ==============================================================================


class RWKV_CMix_x070(nn.Module):
    n_embd: int
    dim_ffn: int
    layer_id: int
    n_layer: int

    def setup(self):
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

        # Time mix parameter
        ddd = jnp.linspace(0, 1, self.n_embd)
        self.x_k = self.param(
            "x_k", lambda k, s: 1.0 - (ddd ** (ratio_1_to_almost0**4)), (self.n_embd,)
        )

        self.key = nn.Dense(
            self.dim_ffn,
            use_bias=False,
            kernel_init=lambda k, s, d: (
                jax.random.uniform(k, s, d, -0.5, 0.5) / math.sqrt(self.n_embd)
            ),
        )
        self.value = nn.Dense(
            self.n_embd, use_bias=False, kernel_init=nn.initializers.zeros
        )

    def __call__(self, x, x_prev):
        # x, x_prev: [Batch, Time, Channel] or [Channel] during scan
        # We assume input is [Batch, Time, Channel] for training

        # Token shift: x_prev is x shifted by 1.
        # In a full sequence training, x_prev is passed in or calculated via rolling.
        xx = x_prev - x
        k = x + xx * self.x_k
        k = jax.nn.relu(self.key(k)) ** 2
        return self.value(k)


class RWKV_TMix_x070(nn.Module):
    n_embd: int
    n_head: int
    head_size: int
    layer_id: int
    n_layer: int

    def setup(self):
        C = self.n_embd
        H = self.n_head
        N = self.head_size

        ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

        # Time mixing factors
        ddd = jnp.linspace(0, 1, C)
        self.x_r = self.param(
            "x_r", lambda k, s: 1.0 - (ddd ** (0.2 * ratio_1_to_almost0)), (C,)
        )
        self.x_w = self.param(
            "x_w", lambda k, s: 1.0 - (ddd ** (0.9 * ratio_1_to_almost0)), (C,)
        )
        self.x_k = self.param(
            "x_k", lambda k, s: 1.0 - (ddd ** (0.7 * ratio_1_to_almost0)), (C,)
        )
        self.x_v = self.param(
            "x_v", lambda k, s: 1.0 - (ddd ** (0.7 * ratio_1_to_almost0)), (C,)
        )
        self.x_a = self.param(
            "x_a", lambda k, s: 1.0 - (ddd ** (0.9 * ratio_1_to_almost0)), (C,)
        )
        self.x_g = self.param(
            "x_g", lambda k, s: 1.0 - (ddd ** (0.2 * ratio_1_to_almost0)), (C,)
        )

        # LoRA parameters generation logic
        # w0 (decay base)
        def init_w0(key, shape):
            n = jnp.arange(C)
            # zigzag
            zigzag = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            base = -6 + 6 * (n / (C - 1)) ** (1 + ratio_0_to_1**0.3)
            return base + 0.5 + zigzag * 2.5

        self.w0 = self.param("w0", init_w0, (C,))

        self.w1 = self.param("w1", nn.initializers.zeros, (C, config.decay_lora))
        self.w2 = self.param("w2", rwkv_orthogonal_init(0.1), (config.decay_lora, C))

        # a0
        def init_a0(key, shape):
            n = jnp.arange(C)
            linear = n / (C - 1) - 0.5
            zigzag = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            return -0.19 + zigzag * 0.3 + linear * 0.4

        self.a0 = self.param("a0", init_a0, (C,))
        self.a1 = self.param("a1", nn.initializers.zeros, (C, config.aaa_lora))
        self.a2 = self.param("a2", rwkv_orthogonal_init(0.1), (config.aaa_lora, C))

        # v0
        def init_v0(key, shape):
            n = jnp.arange(C)
            linear = n / (C - 1) - 0.5
            return 0.73 - linear * 0.4

        self.v0 = self.param("v0", init_v0, (C,))
        self.v1 = self.param("v1", nn.initializers.zeros, (C, config.mv_lora))
        self.v2 = self.param("v2", rwkv_orthogonal_init(0.1), (config.mv_lora, C))

        # Gate
        self.g1 = self.param("g1", nn.initializers.zeros, (C, config.gate_lora))
        self.g2 = self.param("g2", rwkv_orthogonal_init(0.1), (config.gate_lora, C))

        # Other scalars
        n = jnp.arange(C)
        linear = n / (C - 1) - 0.5
        self.k_k = self.param("k_k", lambda k, s: 0.71 - linear * 0.1, (C,))
        self.k_a = self.param(
            "k_a", lambda k, s: 1.0 + jnp.zeros(C) * 0.02, (C,)
        )  # simplified 1.02
        self.r_k = self.param("r_k", lambda k, s: jnp.zeros((H, N)) - 0.04, (H, N))

        # Main projections
        scale = 1.0 / math.sqrt(C)
        self.receptance = nn.Dense(
            C, use_bias=False, kernel_init=nn.initializers.uniform(scale * 0.5)
        )
        self.key = nn.Dense(
            C, use_bias=False, kernel_init=nn.initializers.uniform(scale * 0.05)
        )
        self.value = nn.Dense(
            C, use_bias=False, kernel_init=nn.initializers.uniform(scale * 0.5)
        )
        self.output = nn.Dense(C, use_bias=False, kernel_init=nn.initializers.zeros)

        self.ln_x = nn.GroupNorm(num_groups=H, epsilon=64e-5)

    def wkv7_func(self, r, w, k, v, a, b):
        """
        JAX implementation of the Wind Backstepping operator (v7).
        Args:
            r, w, k, v, a, b: [Batch, Time, Head, HeadSize]
        Returns:
            y: [Batch, Time, Head, HeadSize]
        """
        B, T, H, N = r.shape

        # Define the scan function for a single timestep
        # state shape: [Batch, Head, N, N]
        def scan_fn(state, inputs):
            r_t, w_t, k_t, v_t, a_t, b_t = inputs
            # r,w,k,v,a,b are [Batch, Head, N]

            # w_t is [Batch, Head, N] -> needs to act as diagonal decay
            # state update: S = S * w + S @ a @ b.T + v @ k.T

            # 1. Decay state
            # state: [B, H, N, N]
            # w_t: [B, H, N] -> expand to [B, H, N, 1] for broadcasting across rows
            w_act = w_t[..., None]  # [B, H, N, 1]
            state_w = state * w_act

            # 2. Update via attention (a @ b.T)
            # a_t: [B, H, N], b_t: [B, H, N]
            # We need (S @ a) @ b.T.
            # S @ a -> [B, H, N, N] @ [B, H, N, 1] -> [B, H, N, 1]
            # Then @ b.T

            # Let's be explicit with einsum to avoid confusion
            # Term 1: S @ a @ b.T
            # sa = jnp.einsum('bhij,bhj->bhi', state, a_t) # [B, H, N]
            # sab = jnp.einsum('bhi,bhj->bhij', sa, b_t)   # [B, H, N, N]

            # Optimization: Update rule is S_new = S_old * w + (S_old @ a) @ b.T + v @ k.T
            sa = jnp.einsum("bhij,bhj->bhi", state, a_t)
            sab = jnp.einsum("bhi,bhj->bhij", sa, b_t)

            # Term 2: v @ k.T
            vk = jnp.einsum("bhi,bhj->bhij", v_t, k_t)

            state_next = state_w + sab + vk

            # Output: y = S_next @ r
            y_t = jnp.einsum("bhij,bhj->bhi", state_next, r_t)

            return state_next, y_t

        # Initialize state [Batch, Head, N, N]
        init_state = jnp.zeros((B, H, N, N), dtype=r.dtype)

        # Scan over time
        # Move Time axis to 0 for scan: [Time, Batch, Head, N]
        inputs = (
            r.transpose(1, 0, 2, 3),
            w.transpose(1, 0, 2, 3),
            k.transpose(1, 0, 2, 3),
            v.transpose(1, 0, 2, 3),
            a.transpose(1, 0, 2, 3),
            b.transpose(1, 0, 2, 3),
        )

        _, y = lax.scan(scan_fn, init_state, inputs)

        # y is [Time, Batch, Head, N] -> [Batch, Time, Head, N]
        return y.transpose(1, 0, 2, 3)

    def __call__(self, x, x_prev, v_first_in=None):
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        xx = x_prev - x

        # Interpolate inputs
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Projections
        r = self.receptance(xr)

        # w calculation (softplus + tanh lora)
        w_lora = jnp.tanh(xw @ self.w1) @ self.w2
        w = -jax.nn.softplus(-(self.w0 + w_lora)) - 0.5

        k = self.key(xk)
        v = self.value(xv)

        # v residual connection
        # If v_first is None, it means layer 0, use current v. Else compute mix.
        # Note: In strict JAX/Flax scan over layers, handling 'v_first' state needs care.
        # For simplicity in this demo, we approximate: layer 0 sets v_first, others read it.
        # But 'v' changes over Time.
        # Formula: v = v + (v_first - v) * sigmoid(v0 + xv@v1@v2)

        v_mix_factor = jax.nn.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        if self.layer_id == 0:
            v_first = v
        else:
            # We need v_first passed in from layer 0.
            # In standard RWKV v7, v_first is literally the 'v' vector from the FIRST layer.
            v_first = v_first_in
            v = v + (v_first - v) * v_mix_factor

        # a calculation
        a = jax.nn.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # g calculation
        g = jax.nn.sigmoid(xg @ self.g1) @ self.g2

        # k normalization
        kk = k * self.k_k
        # Normalize kk per head
        kk = kk.reshape(B, T, H, N)
        kk_norm = jnp.linalg.norm(kk, axis=-1, keepdims=True) + 1e-12
        kk = kk / kk_norm
        kk = kk.reshape(B, T, C)

        k = k * (1 + (a - 1) * self.k_a)

        # Prepare for Wind Backstepping (Time Mixing)
        # Reshape to [B, T, H, N]
        r_ = r.reshape(B, T, H, N)
        w_ = jnp.exp(-jnp.exp(w.reshape(B, T, H, N)))  # w is logits for decay rate
        k_ = k.reshape(B, T, H, N)
        v_ = v.reshape(B, T, H, N)
        a_ = a.reshape(B, T, H, N)

        # "kk" is the normalized k used for the 'b' term
        # b = -kk
        # The equation expects inputs to the recurrent update.
        # The update is: S = S*w + S*a*b.T + v*k.T
        # In v7 code: ab = (-kk) @ (kk*a).T
        # So 'b' vector is (-kk).
        # 'a' vector input to kernel is actually (kk*a).

        kk_ = kk.reshape(B, T, H, N)
        b_kernel = -kk_
        a_kernel = kk_ * a_

        x_out = self.wkv7_func(r_, w_, k_, v_, a_kernel, b_kernel)
        x_out = x_out.reshape(B, T, C)

        # Group Norm
        x_out = self.ln_x(x_out)

        # Output gating & projection
        # Additional term from r * k * r_k * v
        r_k_expand = self.r_k.reshape(1, 1, H, N)
        extra = jnp.sum(
            r_.reshape(B, T, H, N) * k_.reshape(B, T, H, N) * r_k_expand,
            axis=-1,
            keepdims=True,
        ) * v_.reshape(B, T, H, N)
        x_out = x_out + extra.reshape(B, T, C)

        x_out = self.output(x_out * g)

        return x_out, v_first


class Block(nn.Module):
    config: RWKVConfig
    layer_id: int

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.att = RWKV_TMix_x070(
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.layer_id,
            self.config.n_layer,
        )
        self.ffn = RWKV_CMix_x070(
            self.config.n_embd, self.config.dim_ffn, self.layer_id, self.config.n_layer
        )
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(epsilon=1e-5)

    def __call__(self, x, x_prev, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        # Attention
        x_ln1 = self.ln1(x)
        # Shift trick for attention x_prev
        if self.layer_id == 0:
            # For layer 0, x_prev for att is just zero padded shift of ln0(x) usually,
            # or the input embeddings shifted.
            # In standard RWKV training:
            # We calculate x_prev for the whole block sequence outside or internally.
            # Here we assume x_prev is passed correctly (shifted x).
            # But wait, LN1 is applied AFTER shift? No, typically shift is on the input to LN.
            # Let's standard RWKV pattern:
            # xx = ln1(x)
            # att(xx, shifted_xx)
            pass

        # Create shifted version of x_ln1 for Attention
        # Note: In efficient impl, we shift the raw x, then LN?
        # RWKV-v4/5/6/7 standard: LN(x), then shift the LN output for mix factors.
        # But 'x' residual is added.

        # Shift logic: [B, T, C] -> shift T right by 1
        x_prev_ln1 = jnp.pad(x_ln1[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

        att_out, v_first_out = self.att(x_ln1, x_prev_ln1, v_first)
        x = x + att_out

        # FFN
        x_ln2 = self.ln2(x)
        x_prev_ln2 = jnp.pad(x_ln2[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

        ffn_out = self.ffn(x_ln2, x_prev_ln2)
        x = x + ffn_out

        return x, v_first_out


class RWKV7(nn.Module):
    config: RWKVConfig

    def setup(self):
        self.emb = nn.Embed(
            self.config.vocab_size,
            self.config.n_embd,
            embedding_init=nn.initializers.uniform(1e-4),
        )

        self.blocks = [Block(self.config, i) for i in range(self.config.n_layer)]

        self.ln_out = nn.LayerNorm(epsilon=1e-5)
        self.head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=rwkv_orthogonal_init(0.5),
        )

    def __call__(self, idx):
        # idx: [Batch, Time]
        x = self.emb(idx)

        v_first = None

        for block in self.blocks:
            # Note: We handle x_prev (time shift) inside the block logic relative to LN
            # Here we just pass x. The Block generates its own shifted views of LN(x).
            # v_first flows through layers (generated by layer 0, used by others)
            x, v_first = block(
                x, None, v_first
            )  # x_prev arg is handled inside block now via padding

        x = self.ln_out(x)
        x = self.head(x)
        return x


# ==============================================================================
# 4. Data Preparation (Synthetic)
# ==============================================================================


def get_batch(config, key):
    """
    Generates a batch of synthetic data.
    Task: Digit copying / Reversal / Simple pattern.
    Format: [Num, Num, ..., Separator, Num, Num, ...]
    """
    B, T = config.batch_size, config.ctx_len

    # Simple task: Repeat sequence
    # Input:  [A, B, C, SEP, A, B, C]
    # Target: [B, C, SEP, A, B, C, END]

    k1, k2 = random.split(key)
    vocab_subset = 100  # use first 100 tokens

    data = random.randint(k1, (B, T // 2), 0, vocab_subset)
    sep = jnp.full((B, 1), vocab_subset, dtype=jnp.int32)  # Token 100 is separator

    inputs = jnp.concatenate([data, sep, data], axis=1)  # [B, T+1]
    inputs = inputs[:, :T]  # trim to ctx_len

    targets = jnp.roll(inputs, -1, axis=1)

    # Simple mask to not train on the first half (optional, but standard usually trains on all)
    # For simplicitly, standard causal language modeling on the whole sequence.
    return jnp.array(inputs, dtype=jnp.int32), jnp.array(targets, dtype=jnp.int32)


# ==============================================================================
# 5. Training Loop
# ==============================================================================


def create_train_state(rng, config):
    model = RWKV7(config)
    dummy_input = jnp.ones((1, config.ctx_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)["params"]

    # Cosine decay schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr_init,
        warmup_steps=10,
        decay_steps=config.total_steps,
        end_value=config.lr_final,
    )

    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.1)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


@jax.jit
def train_step(state, batch_idx, batch_targets):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_idx)
        # Cross Entropy Loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_targets)
        return loss.mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    print(f"Initializing RWKV-v7 (JAX) on {jax.devices()}...")

    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)

    state = create_train_state(init_rng, config)

    # Parameter count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model Parameters: {param_count/1e6:.2f}M")

    print("Starting training...")
    start_time = time.time()

    # Re-create schedule for logging
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr_init,
        warmup_steps=10,
        decay_steps=config.total_steps,
        end_value=config.lr_final,
    )

    for step in range(config.total_steps):
        rng, batch_rng = random.split(rng)
        inputs, targets = get_batch(config, batch_rng)

        state, loss = train_step(state, inputs, targets)

        if step % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (config.batch_size * config.ctx_len * (step + 1)) / elapsed
            print(
                f"Step {step:04d} | Loss: {loss:.4f} | LR: {lr_schedule(state.step).item():.2e} | Tok/s: {tokens_per_sec:.0f}"
            )

    print("Training finished.")

    # Simple Inference Check
    print("\nInference Check (Prompt completion):")
    prompt = jnp.array([[1, 2, 3, 100]], dtype=jnp.int32)  # 1,2,3 SEP
    # Expect 1, 2, 3...

    curr_ids = prompt
    print(f"Prompt: {curr_ids}")

    # Dumb autoregressive generation loop
    for _ in range(5):
        logits = state.apply_fn({"params": state.params}, curr_ids)
        next_token = jnp.argmax(logits[0, -1, :], axis=-1)
        curr_ids = jnp.concatenate([curr_ids, next_token[None, None]], axis=1)

    print(f"Gen:    {curr_ids}")


if __name__ == "__main__":
    main()
