# Andromeda

A 63M-parameter GPT-style language model, trained from scratch on a single laptop in 6.3 hours.

> **WASHINGTON** — The Senate voted today to take a vote on the measure of a $50 million bill for an effort to prevent the sale of marijuana.
>
> "The president's proposal is not in place to get the legal counsel that he has made, and it will be taken down by Congress," said Rep. Paul Durbin, R-Texas.

*— unedited output from Andromeda, given only the prompt "WASHINGTON — The Senate voted today to"*

---

## What this is

Andromeda is a decoder-only transformer language model, built and trained entirely from scratch. No fine-tuning, no pre-trained weights — the model started from random initialisation and learned English from 509M tokens of web text over 30,000 training steps.

It's not a frontier model. It's deliberately small, trained on modest hardware, to demonstrate end-to-end competence across the full LLM pipeline: data ingestion, tokenisation, model architecture, training, inference, evaluation. Every component was written by hand in PyTorch.

This project is the production-scale counterpart to [Forge](https://github.com/YOUR_USERNAME/Forge), a from-scratch ML framework I built previously. Forge demonstrated understanding of ML internals by reimplementing them; Andromeda demonstrates the ability to execute at scale using production tooling.

## Technical highlights

- **Model**: 63.49M-parameter decoder-only transformer (8 layers, 8 heads, 512 embedding dim, 512 context length)
- **Training**: 30,000 steps of AdamW with warmup + cosine LR decay, gradient clipping, per-parameter-group weight decay
- **Architecture choices**: weight-tied embeddings, pre-norm residual blocks, GELU feed-forward, no biases in Linear/LayerNorm (LLaMA convention)
- **Data pipeline**: OpenWebText tokenised with GPT-2 BPE (tiktoken), parallelised across 12 CPU cores, memory-mapped binary files for zero-copy batch sampling
- **Training stability**: GPT-2-style residual projection initialisation scaled by 1/√(2·n_layer) to prevent variance drift through deep residual streams
- **Inference**: top-k + top-p (nucleus) sampling with repetition penalty over a rolling 64-token window
- **Hardware**: single Apple M4 Max (unified memory, MPS backend via PyTorch)

## Results

![Validation loss and perplexity over training](loss_curve.png)

**Final validation loss: 4.21** (perplexity 67.3) after 6 hours 21 minutes of training.

| Step   | Val loss | Perplexity |
|--------|----------|------------|
| 0      | 10.93    | 55,960     |
| 5,000  | 5.28     | 196        |
| 10,000 | 4.76     | 117        |
| 15,000 | 4.49     | 89         |
| 20,000 | 4.34     | 77         |
| 25,000 | 4.25     | 70         |
| 30,000 | **4.21** | **67**     |

For reference: a fully-trained GPT-2 Small (124M params, 40B tokens of training data) reaches perplexity ~30-40 on OpenWebText. Andromeda closes roughly a third of that gap with ½ the parameters and ~1/160th of the training data.

### Training throughput

Sustained ~11,150 tokens/sec on an M4 Max throughout the 6-hour run. No divergence, no crashes, no thermal emergencies. Each training step processed 8,192 tokens (batch size 16 × sequence length 512).

## Sample outputs

All samples produced with `temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.3`, unedited.

### News prompt

> **Prompt**: *"WASHINGTON — The Senate voted today to"*
>
> **Output**: *"take a vote on the measure of a $50 million bill for an effort to prevent the sale of marijuana. 'The president's proposal is not in place to get the legal counsel that he has made, and it will be taken down by Congress,' said Rep. Paul Durbin, R-Texas."*

The model has clearly internalised journalistic conventions: dateline format, lead with dollar figure, quoted statement with proper attribution (`"said Rep. X, R-State"`). The specific facts (`Paul Durbin` instead of `Dick Durbin`; wrong state) reflect undertrained world knowledge.

### Corporate PR prompt

> **Prompt**: *"A spokesperson for the company said"*
>
> **Output**: *"in a statement, 'Our customers are very happy about the upcoming launch of our new product. We want to provide this unique experience that will make it an opportunity to bring more people into the market.'"*

Reads as boilerplate corporate communications — an impressive match for the PR-release style common in OpenWebText.

### Narrative prompt

> **Prompt**: *"Once upon a time"*
>
> **Output**: *"I could easily see how the weather was going on in New York. The rain, too, had been about two-and-a-half miles south of New York City. That meant that people would be walking out to the mountains to get a place off the ground and all that was there were huge roads."*

First-person narrative with spatial reasoning (cardinal directions, distance measurements). OpenWebText has relatively little fairy-tale content, so the model defaulted to the most common first-person genre in its training data.

## Failure modes

Being honest about where the model fails is as important as showing where it succeeds.

### Semantic inconsistency
The model produces grammatical sentences with contradictory or nonsensical content. *"The capital of France is a small, huge, massive-scale and complex city"* — every word is a plausible English word, every pair flows grammatically, but the sentence is meaningless. This is the expected failure mode of an undertrained language model that has learned phrase shapes without learning what they refer to.

### Topic drift over long generations
Samples that start on one topic often migrate to another after 50-100 tokens. This reflects the 512-token context window and limited training data — the model has learned what different genres look like locally but hasn't seen enough text to learn *when* to switch.

### Fragile factual retrieval
Simple factual questions fail unpredictably. Prompted with "The capital of France is", the model produces generic news-about-France output without ever confidently emitting "Paris". World knowledge requires orders of magnitude more training data than this model received.

### Repetition without safeguards
Without the repetition penalty, low-temperature generation collapses into loops (e.g. "the crisis has been a major problem ..." × 20). The repetition penalty applied to the last 64 tokens eliminates this almost entirely. This is a known failure mode of small LMs and a well-understood problem with a standard solution.

## Architecture

```
Input tokens (B, T)
    │
    ├─▶ Token Embedding (50,257 × 512)
    └─▶ Positional Embedding (512 × 512)
        │
        ▼
    [ Transformer Block ] × 8
    │
    │  ┌──────────────────────────────────┐
    │  │ x = x + MultiHeadAttn(LN(x))      │   ← pre-norm residual
    │  │ x = x + FeedForward(LN(x))        │
    │  │                                   │
    │  │  MultiHeadAttn: 8 heads × 64 dim  │
    │  │  FeedForward:  512 → 2048 → 512   │
    │  │                  with GELU         │
    │  └──────────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Language-model head (weight-tied to embedding)
    │
    ▼
Logits (B, T, 50,257)
```

Key design decisions:
- **Weight tying** between token embedding and LM head, reclaiming ~26M parameters
- **Pre-norm** residual connections (modern LLM convention; more stable than post-norm at scale)
- **No biases** in Linear or LayerNorm (LLaMA convention; marginally improves training)
- **Scaled residual init** with std `0.02 / sqrt(2·n_layer)` on projection layers, preventing residual stream variance from exploding in deep networks

## Training setup

| Hyperparameter         | Value             |
|------------------------|-------------------|
| Optimiser              | AdamW             |
| Betas                  | (0.9, 0.95)       |
| Weight decay           | 0.1 (matrices only) |
| Peak learning rate     | 3e-4              |
| Min learning rate      | 3e-5              |
| Warmup steps           | 600               |
| Total steps            | 30,000            |
| LR schedule            | Linear warmup + cosine decay |
| Gradient clipping      | 1.0 (L2 norm)     |
| Batch size             | 16 sequences      |
| Sequence length        | 512 tokens        |
| Effective tokens/step  | 8,192             |
| Total tokens seen      | 245,760,000       |

Validation loss evaluated every 1,000 steps on 50 randomly-sampled batches from a held-out split.

## Repository structure

```
Andromeda/
├── data/
│   ├── download.py       # parallelised tokenisation of OpenWebText
│   └── dataloader.py     # mmap-based random-window batch sampling
├── model/
│   ├── config.py         # GPTConfig dataclass + preset sizes
│   ├── layers.py         # LayerNorm, FeedForward, TokenAndPositionEmbedding
│   ├── attention.py      # Multi-head causal self-attention w/ Flash Attention
│   └── transformer.py    # TransformerBlock + full GPT model
├── training/
│   ├── optimizer.py      # AdamW with split weight decay + LR schedule
│   ├── utils.py          # Checkpointing + timing
│   └── train.py          # Main training loop
├── evaluate/
│   ├── generate.py       # top-k + top-p sampling w/ repetition penalty
│   └── plot_loss.py      # Loss-curve visualisation from training logs
├── checkpoints/
│   └── medium/best.pt    # trained 63M-param checkpoint
└── requirements.txt
```

## How to run

```bash
# Set up
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Prepare data (downloads OpenWebText, tokenises in parallel)
python -m data.download

# Train
python -m training.train

# Generate
python -m evaluate.generate \
    --ckpt checkpoints/medium/best.pt \
    --prompt "The future of" \
    --temperature 0.8 --top_k 40 --top_p 0.9 --repetition_penalty 1.3
```

## What would improve this further

Specific next steps, in rough order of return on investment:

1. **More training data and steps**: the loss curve was still descending at step 30K. Another 50K steps on a 2B-token dataset would likely reach val loss ~3.8.
2. **Mixed precision (bfloat16)**: 1.5-2× speedup on M4 Max with no quality loss, allowing larger batches within the same memory budget.
3. **Rotary position embeddings (RoPE)**: replace learned positional embeddings with rotary encodings. Standard in every modern LLM (LLaMA, Mistral, Claude).
4. **SwiGLU feed-forward**: replace GELU-based FFN with gated variant; ~10-15% quality improvement at same parameter count.
5. **Domain fine-tuning**: continued training on a narrower, cleaner dataset (e.g. Wikipedia) would produce markedly better quality in that domain.

## Acknowledgements

This project draws on the lineage of open-source work on small transformer training: Karpathy's nanoGPT for the overall structure, the LLaMA papers for architecture choices, and the OpenWebText dataset maintained by the EleutherAI community. The training philosophy — that small, well-run experiments teach more than frontier models ever can — comes directly from Karpathy's writing.

---

Built by [Srihari](https://github.com/SrihariSr), 2026.