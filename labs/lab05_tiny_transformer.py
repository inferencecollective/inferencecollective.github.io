"""Lab 05: Causal Self-Attention, Tiny Transformer and Safe Checkpointing in PyTorch."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from labs.common import DATA, OUTPUTS, PREVIEWS


class TinyCausalTransformer(nn.Module):
    """Minimal autoregressive Transformer block with causal self-attention."""
    def __init__(self, vocab_size=12, embed_dim=16, num_heads=2, max_len=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn_out, attn_weights = self.attn(h, h, h, attn_mask=causal_mask)
        h = self.ln1(h + attn_out)
        ffn_out = self.ffn(h)
        h = self.ln2(h + ffn_out)
        logits = self.head(h)
        return logits, attn_weights


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    data_file = DATA / "pytorch-sequences.pt"
    if not data_file.exists():
        from data.generate_samples import generate_all
        generate_all()

    payload = torch.load(str(data_file), weights_only=True)
    sequences = payload["sequences"]
    vocab_size = payload.get("vocab_size", 12)
    seq_len = payload.get("seq_len", 8)

    # Inputs: tokens [0 .. T-2], Targets: tokens [1 .. T-1]
    X_all = sequences[:, :-1]
    Y_all = sequences[:, 1:]

    # 1. Train / Test split (240 train, 60 test)
    torch.manual_seed(42)
    n_samples = len(sequences)
    indices = torch.randperm(n_samples)
    n_train = 240
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_test, Y_test = X_all[test_idx], Y_all[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=24, shuffle=True)

    # 2. Model Initialization & Training
    model = TinyCausalTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, max_len=seq_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)

    train_losses, test_losses = [], []
    epochs = 30

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            # logits: (B, T, V) -> (B*T, V), targets: (B, T) -> (B*T)
            loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)

        epoch_loss = running_loss / len(X_train)

        model.eval()
        with torch.inference_mode():
            test_logits, _ = model(X_test)
            test_loss = criterion(test_logits.reshape(-1, vocab_size), Y_test.reshape(-1)).item()

        train_losses.append(epoch_loss)
        test_losses.append(test_loss)

    # Calculate token-level test accuracy
    model.eval()
    with torch.inference_mode():
        sample_logits, sample_attn = model(X_test)
        pred_tokens = sample_logits.argmax(dim=-1)
        test_acc = (pred_tokens == Y_test).float().mean().item()

    # 3. Safe Checkpointing Demonstration
    ckpt_path = OUTPUTS / "05-tiny-transformer.pt"
    torch.save(model.state_dict(), str(ckpt_path))

    # Verify reloading with weights_only=True
    restored_model = TinyCausalTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, max_len=seq_len)
    state_dict = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    restored_model.load_state_dict(state_dict)

    weights_match = True
    for p1, p2 in zip(model.parameters(), restored_model.parameters()):
        if not torch.allclose(p1, p2):
            weights_match = False
            break
    assert weights_match, "Restored model weights did not match original model"

    # 4. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")

    # (0, 0) Causal Attention Heatmap (showing strict lower-triangular visibility)
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    attn_matrix = sample_attn[0].detach().cpu().numpy()  # (T, T)
    im0 = ax0.imshow(attn_matrix, cmap="inferno")
    ax0.set_title("1. Causal Self-Attention Weights (Lower Triangular)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Key Position (Source)", color="#94a3b8", fontsize=9)
    ax0.set_ylabel("Query Position (Target)", color="#94a3b8", fontsize=9)
    ax0.tick_params(colors="#94a3b8")
    for spine in ax0.spines.values():
        spine.set_color("#334155")
    # Annotate values
    T = attn_matrix.shape[0]
    for i in range(T):
        for j in range(T):
            if j <= i:
                ax0.text(j, i, f"{attn_matrix[i, j]:.2f}", ha="center", va="center", color="#f8fafc", fontsize=8)

    # (0, 1) Cross-Entropy Loss Curve
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.plot(train_losses, color="#38bdf8", linewidth=2, label="Train Loss")
    ax1.plot(test_losses, color="#fde047", linewidth=2, linestyle="--", label="Test Loss")
    ax1.set_title(f"2. Next-Token CE Loss (Final Test: {test_losses[-1]:.4f})", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("Cross-Entropy Loss", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # (1, 0) Autoregressive Next-Token Rollout
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    sample_idx = 0
    in_seq = X_test[sample_idx].numpy()
    target_seq = Y_test[sample_idx].numpy()
    pred_seq = pred_tokens[sample_idx].numpy()

    positions = np.arange(len(in_seq))
    ax2.plot(positions, in_seq, color="#94a3b8", marker="o", label="Input Tokens [t]")
    ax2.plot(positions, target_seq, color="#34d399", marker="^", label="True Next Tokens [t+1]")
    ax2.plot(positions, pred_seq, color="#f43f5e", marker="x", linestyle="--", label="Predicted Tokens [t+1]")
    ax2.set_title(f"3. Next-Token Sequence Rollout (Test Acc: {test_acc*100:.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sequence Index t", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Token ID", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Checkpoint Integrity Audit
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    ax3.set_title("4. Safe Checkpoint Audit (weights_only=True)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.axis("off")

    num_layers = len(list(model.state_dict().keys()))
    total_weights = sum(p.numel() for p in model.parameters())
    audit_lines = [
        f"✓ Checkpoint Path: {ckpt_path.name}",
        f"✓ Serialization: torch.save(state_dict)",
        f"✓ Ingestion: torch.load(weights_only=True)",
        f"✓ Total Parameters: {total_weights:,}",
        f"✓ Registered State Keys: {num_layers}",
        f"✓ Round-trip allclose Verification: PASS",
        f"✓ Attack Vector Mitigation: Arbitrary Code Execution Blocked"
    ]
    for idx, line in enumerate(audit_lines):
        color_line = "#34d399" if "✓" in line else "#94a3b8"
        ax3.text(0.04, 0.85 - idx * 0.12, line, color=color_line, fontsize=9.5, fontweight="bold", transform=ax3.transAxes)

    plt.suptitle("PyTorch Lab 05: Causal Transformer & Safe Checkpoint Audit", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "05-causal-attention.png"
    preview_file = PREVIEWS / "05-causal-attention.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 05 figure to {out_file} and checkpoint to {ckpt_path}")
    return {
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_losses[-1]),
        "weights_verified": bool(weights_match),
        "checkpoint_path": str(ckpt_path),
        "total_parameters": int(total_weights),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
