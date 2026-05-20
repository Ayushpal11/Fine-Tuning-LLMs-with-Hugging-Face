# 🦙 Fine-Tuning LLaMA 2 with Hugging Face on Google Colab

Fine-tune a LLaMA 2 model on medical terminology using QLoRA (4-bit quantization + LoRA adapters) — runnable on a free Colab T4 GPU.

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| Platform | Google Colab (free tier works) |
| GPU | T4 or better (**required** — CPU will not work) |
| Python | 3.10+ |
| Disk | ~6 GB for model weights |

> ⚠️ **Before running:** Go to `Runtime → Change runtime type → T4 GPU` in Colab.  
> After installing packages in Step 1, **restart the runtime** before proceeding.

---

## 📦 Package Versions

These versions are pinned for mutual compatibility:

| Package | Version |
|---|---|
| `torch` | 2.2.0 |
| `transformers` | 4.36.2 |
| `peft` | 0.7.1 |
| `trl` | 0.7.4 |
| `bitsandbytes` | 0.41.3 |
| `accelerate` | 0.26.1 |
| `datasets` | 2.16.1 |
| `huggingface_hub` | 0.20.3 |

---

## 🗂️ Notebook Structure

### Step 1 — Install Packages
Installs all dependencies with pinned versions. Run this cell first, then **restart the runtime**.

### Step 2 — Import Libraries
Loads PyTorch, Hugging Face Transformers, PEFT, TRL, and related libraries.

### Step 3 — Load the Model
Loads [`aboonaji/llama2finetune-v2`](https://huggingface.co/aboonaji/llama2finetune-v2) in **4-bit NF4 quantization** via `BitsAndBytesConfig`, reducing VRAM usage from ~14 GB to ~4 GB.

### Step 4 — Load the Tokenizer
Loads the matching tokenizer with `eos_token` as the padding token and right-side padding.

### Step 5 — Training Arguments
Configures the Hugging Face `TrainingArguments`:
- `fp16=True` — mixed precision for speed
- `paged_adamw_8bit` — memory-efficient optimizer
- `max_steps=100` — short run; increase for better results
- `report_to="none"` — disables wandb (remove to re-enable)

### Step 6 — SFT Trainer
Creates a `SFTTrainer` with:
- **Dataset:** [`aboonaji/wiki_medical_terms_llam2_format`](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format)
- **LoRA config:** rank `r=64`, alpha `16`, dropout `0.1`
- **Max sequence length:** 512 tokens

### Step 7 — Train
Runs the fine-tuning loop. Logs loss every 10 steps.

### Step 8 — Inference
Runs a sample prompt through the fine-tuned model using a `text-generation` pipeline:
```
<s>[INST] Please tell me about Bursitis [/INST]
```

---

## 🧠 Architecture Overview

```
LLaMA 2 Base (aboonaji/llama2finetune-v2)
        │
        ▼
 4-bit NF4 Quantization (bitsandbytes)
        │
        ▼
   LoRA Adapters (PEFT)
   r=64, alpha=16
        │
        ▼
  SFT on Medical Terms Dataset
        │
        ▼
  Fine-Tuned Medical LLaMA 2
```

---

## ⚙️ Key Configuration

```python
# Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# LoRA
LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=16,
    lora_dropout=0.1
)
```

---

## 🔧 Customization Tips

- **More training:** Increase `max_steps` (e.g. `500` or `1000`) for better convergence
- **Different dataset:** Replace `aboonaji/wiki_medical_terms_llam2_format` with any Hugging Face dataset; update `dataset_text_field` to match
- **Reduce memory:** Lower `per_device_train_batch_size` to `1` or `2`, or reduce `max_seq_length` to `256`
- **Save the model:** Add `llama_sft_trainer.save_model("./my-model")` after training
- **Enable wandb logging:** Remove `report_to="none"` and log in at [wandb.ai/authorize](https://wandb.ai/authorize)

---

## ❓ Common Issues

| Error | Fix |
|---|---|
| `RuntimeError: No GPU found` | Enable GPU: `Runtime → Change runtime type → T4 GPU` |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to `1`, lower `max_seq_length` to `256` |
| Import errors after install | Restart the runtime after Step 1 |
| Slow training | Normal on free Colab — 100 steps takes ~5–10 minutes on T4 |

---

## 📄 License

This notebook is for educational purposes. Model weights are subject to the [LLaMA 2 Community License](https://ai.meta.com/llama/license/).
