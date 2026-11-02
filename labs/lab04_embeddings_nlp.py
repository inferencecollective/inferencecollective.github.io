"""Lab 04: Token Embeddings, Masked Pooling and Toy NLP Classification in PyTorch."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from labs.common import DATA, OUTPUTS, PREVIEWS


class TextEmbeddingClassifier(nn.Module):
    """Sequence classifier using learned token embeddings and masked mean pooling."""
    def __init__(self, vocab_size=16, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x, use_mask=True):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, D)
        if use_mask:
            mask = (x != 0).unsqueeze(-1)  # (B, T, 1)
            summed = (emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # (B, D)
        else:
            pooled = emb.mean(dim=1)  # Naive unmasked mean
        return self.fc(pooled)


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    data_file = DATA / "pytorch-corpus.pt"
    if not data_file.exists():
        from data.generate_samples import generate_all
        generate_all()

    payload = torch.load(str(data_file), weights_only=True)
    tokens = payload["tokens"]
    labels = payload["labels"]
    vocab = payload["vocab"]
    id_to_word = {v: k for k, v in vocab.items()}

    # 1. Train / Test split (150 train, 50 test)
    torch.manual_seed(42)
    n_samples = len(tokens)
    indices = torch.randperm(n_samples)
    n_train = 150
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    tok_train, lbl_train = tokens[train_idx], labels[train_idx]
    tok_test, lbl_test = tokens[test_idx], labels[test_idx]

    train_loader = DataLoader(TensorDataset(tok_train, lbl_train), batch_size=16, shuffle=True)

    # 2. Model Training
    model = TextEmbeddingClassifier(vocab_size=len(vocab), embed_dim=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    epochs = 35
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x, use_mask=True)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            correct_train += (logits.argmax(dim=-1) == batch_y).sum().item()

        epoch_train_loss = running_loss / len(tok_train)
        epoch_train_acc = correct_train / len(tok_train)

        model.eval()
        with torch.inference_mode():
            test_logits = model(tok_test, use_mask=True)
            test_loss = criterion(test_logits, lbl_test).item()
            test_acc = (test_logits.argmax(dim=-1) == lbl_test).float().mean().item()

        train_losses.append(epoch_train_loss)
        test_losses.append(test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(test_acc)

    final_test_acc = test_accs[-1]

    # 3. Embedding Analysis: 2D PCA of learned 8D vectors
    model.eval()
    with torch.inference_mode():
        emb_weights = model.embedding.weight.detach()  # (16, 8)
        # SVD for PCA
        u, s, v = torch.pca_lowrank(emb_weights, q=2)
        pca_2d = torch.matmul(emb_weights, v[:, :2]).numpy()

    # 4. Compare Masked vs Unmasked Mean Pooling
    # Create test vectors with increasing padding tokens (0, 1, 2, 3, 4 pad tokens)
    pad_counts = [0, 1, 2, 3, 4]
    norm_masked, norm_naive = [], []
    sample_seq_base = torch.tensor([2, 4, 6, 3, 5, 7], dtype=torch.long)
    for p_cnt in pad_counts:
        seq = sample_seq_base.clone()
        if p_cnt > 0:
            seq[-p_cnt:] = 0
        seq_batch = seq.unsqueeze(0)
        with torch.inference_mode():
            emb = model.embedding(seq_batch)
            # Masked
            mask = (seq_batch != 0).unsqueeze(-1)
            p_masked = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # Naive
            p_naive = emb.mean(dim=1)
            norm_masked.append(torch.norm(p_masked).item())
            norm_naive.append(torch.norm(p_naive).item())

    # 5. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")

    # (0, 0) 2D Embedding Space Projection
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    for tid, word in id_to_word.items():
        x_coord, y_coord = pca_2d[tid, 0], pca_2d[tid, 1]
        if tid == 0:
            color = "#64748b"  # <PAD>
        elif tid in [2, 3, 4, 5, 6, 7]:
            color = "#34d399"  # Positive
        elif tid in [8, 9, 10, 11, 12, 13]:
            color = "#f43f5e"  # Negative
        else:
            color = "#38bdf8"  # Neutral / UNK
        ax0.scatter(x_coord, y_coord, color=color, s=80, edgecolor="#0f172a")
        ax0.text(x_coord + 0.04, y_coord + 0.04, word, color=color, fontsize=9.5, fontweight="bold")
    ax0.set_title("1. Learned Token Embeddings (2D PCA Semantic Projection)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.tick_params(colors="#94a3b8")
    for spine in ax0.spines.values():
        spine.set_color("#334155")

    # (0, 1) Masked vs Naive Mean Pooling
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.plot(pad_counts, norm_masked, color="#34d399", marker="o", linewidth=2.5, label="Masked Mean (Invariant to PAD)")
    ax1.plot(pad_counts, norm_naive, color="#f43f5e", marker="s", linestyle="--", linewidth=2, label="Naive Mean (Diluted by PAD)")
    ax1.set_title("2. Sentence Embedding Norm vs Padding Tokens", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of <PAD> Tokens Appended", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("Pooled Vector L2 Norm", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # (1, 0) Training & Validation Loss
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    ax2.plot(train_losses, color="#38bdf8", linewidth=2, label="Train Loss")
    ax2.plot(test_losses, color="#fde047", linewidth=2, linestyle="--", label="Test Loss")
    ax2.set_title(f"3. Training Convergence (Final Test Acc: {final_test_acc*100:.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Sample Inference Sequences
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    sample_tokens = tok_test[:6].numpy()
    with torch.inference_mode():
        sample_probs = torch.softmax(model(tok_test[:6]), dim=-1)[:, 1].numpy()

    ax3.set_title("4. Sample Sequence Predictions (P(Pos))", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.axis("off")
    for idx in range(6):
        words = [id_to_word[t] for t in sample_tokens[idx]]
        prob_pos = sample_probs[idx]
        true_lbl = lbl_test[idx].item()
        lbl_str = "POS" if true_lbl == 1 else "NEG"
        pred_str = "POS" if prob_pos > 0.5 else "NEG"
        color_res = "#34d399" if lbl_str == pred_str else "#f43f5e"
        y_pos = 0.85 - idx * 0.15
        text_str = f"Seq {idx+1}: [{', '.join(words)}] -> True: {lbl_str} | Pred: {pred_str} ({prob_pos*100:.1f}%)"
        ax3.text(0.02, y_pos, text_str, color=color_res, fontsize=9, fontweight="bold", transform=ax3.transAxes)

    plt.suptitle("PyTorch Lab 04: Token Embeddings & Masked Mean Pooling", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "04-embeddings-nlp.png"
    preview_file = PREVIEWS / "04-embeddings-nlp.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 04 figure to {out_file} and {preview_file}")
    return {
        "final_test_acc": float(final_test_acc),
        "vocab_size": len(vocab),
        "embed_dim": 8,
        "epochs": epochs,
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
