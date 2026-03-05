 # TTS Fine-Tuning Project (Unsloth CSM-1B + LoRA)

> **Goal:** build a small, reproducible pipeline that (1) prepares a paired *text ↔ audio* dataset and (2) fine-tunes a TTS model using LoRA, then (3) runs inference (including “voice-style prompting”) on the fine-tuned checkpoint.

---

## 1) Project Summary

This project fine-tunes the **`unsloth/csm-1b`** model using **Unsloth’s `FastModel`** wrapper and a **LoRA** adapter (rank `r=32`) for efficient training.

What’s in the notebook:

- Loading the dataset from Hugging Face: `load_dataset("Nunsi/tts-data", split="train")`  
- Casting the `audio` column to Hugging Face’s `Audio` feature at **24 kHz**
- Ensuring a **single-speaker** setup via a default `source="0"` column (speaker id)
- Preparing model inputs with `processor.apply_chat_template(...)`
- Training with `transformers.Trainer` and safe memory settings
- Inference examples (with and without a “voice prompt” audio context)
- Saving checkpoints locally and (optionally) pushing to Hugging Face

**Primary artifact:** `TTS_Model.ipynb` (evidence: cells 7, 9, 14, 15, 19, 20, 26–28, 30–40).

---

## 2) Repository Layout (recommended)

Even if you run everything in Colab, having a clear structure helps reviewers reproduce your work:

```
.
├─ TTS_Model.ipynb
├─ scripts/
│  ├─ chunk_audio_30s.py          # fixed-window chunking (audio-only)
│  ├─ build_hf_dataset.py         # builds rows + pushes to hub
│  └─ utils.py
├─ data/
│  ├─ audio_clean/               # cleaned long-form wav files
│  ├─ transcripts/               # Whisper JSON outputs (word timestamps)
│  └─ clips/                     # exported 30s wav clips
├─ outputs/                      # Trainer outputs
├─ logs/                         # TensorBoard logs
└─ requirements.txt              # exported from pip freeze
```

**Note:** Your notebook writes outputs to `/content/...` (Colab default). In a repo, you can map those to `data/...`.

---

## 3) Dataset Preparation

Dataset used in training notebook:
  
https://huggingface.co/datasets/Nunsi/tts-data
---

## 4) Model + Training Details (from `TTS_Model.ipynb`)

### 4.1 Base model

- Base: `unsloth/csm-1b`
- Loader: `FastModel.from_pretrained(...)`
- Auto model class: `CsmForConditionalGeneration`  
(evidence: cell 7)

### 4.2 LoRA configuration

You fine-tune with LoRA:

- `r = 32`
- target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `lora_alpha = 16`, `lora_dropout = 0`
- gradient checkpointing: `"unsloth"`
- rank-stabilized LoRA: `use_rslora = True`  
(evidence: cell 9)

### 4.3 Audio decoding stability fix (Colab / T4)

The notebook applies a stability fix by:

- setting `os.environ["HF_AUDIO_CODEC"] = "soundfile"` **before** importing `datasets`
- uninstalling `torchcodec` if present (it can crash on certain setups)
- ensuring `soundfile` and `librosa` are installed  
(evidence: cell 14)

### 4.4 Preprocessing

You build a “conversation” structure per example and call:

- `processor.apply_chat_template(..., output_labels=True, ...)`

You also enforce safe limits:

- `max_text_length = 500`
- `max_audio_length = 480_000` (≈20 seconds at 24 kHz)

And you added **manual truncation** to guarantee max length before feeding the processor (evidence: cell 19).

### 4.5 Training configuration

Trainer highlights (evidence: cell 20):

- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 1` (to avoid stacking/shape issues)
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 2`
- `learning_rate = 2e-4`
- `optim = "adamw_8bit"`
- `report_to = "tensorboard"`
- `dataloader_drop_last = True` (to avoid last-batch shape mismatch)

You also log GPU memory stats before and after training (evidence: cells 22 and 24).

---

## 5) Inference Modes

The notebook includes:

### 5.1 Generation without context (“zero-shot style”)

You tokenize a prompt like:

- `inputs = processor(f"[speaker_id]{text}", ...)`
- `audio_values = model.generate(..., output_audio=True)`

Then save a wav and compare to a real sample (evidence: cell 26).

### 5.2 Generation with context (“voice-style prompting”)

You provide a *reference* text+audio pair first, then the target text.

Conceptually:
1) Reference: reference transcript + reference audio  
2) Target: desired text only  

This encourages the model to follow the reference voice/style (evidence: cell 27).

### 5.3 Interactive loop

There’s an interactive cell that repeatedly:
- accepts user text
- generates audio
- saves outputs with timestamps  
(evidence: cell 28)

---

## 6) Saving / Exporting

### 6.1 Save locally

You save both model and processor locally:

- `model.save_pretrained(...)`
- `processor.save_pretrained(...)`  
(evidence: cell 30)

### 6.2 Hugging Face push (model)

The notebook includes commented lines showing the intended push flow (log in, then `model.push_to_hub(...)`, `processor.push_to_hub(...)`) (evidence: cells 35–39).

### 6.3 Requirements

You export `requirements.txt` via:

- `pip freeze > requirements.txt`  
(evidence: cell 40)

---

## 7) Research Fellowship Write-up Template (Tailor to the Fatima Institute Call)

I **cannot confirm** the institute’s current fellowship requirements from this environment (my web lookup tool failed during this session), so treat this section as a *template* you can tailor to their actual call.

Suggested framing:

1. **Problem statement:** adapting TTS to a new domain/voice with limited data.
2. **Data contribution:** reproducible pipeline: transcription → fixed-window chunking → HF dataset publication.
3. **Method:** LoRA fine-tuning of a compact base model (`unsloth/csm-1b`) with explicit stability safeguards.
4. **Reproducibility:** notebook + exported requirements + dataset hosted on HF.
5. **Impact:** low-resource speech adaptation, rapid prototyping, deployment readiness.

If you paste the fellowship call link/text, I can rewrite this section to match their scoring rubric.

---

## 8) Ethical / Legal Notes (important for fellowships)

- **Data rights:** only train on audio you have the right to use for ML training.
- **Voice cloning & consent:** if the dataset reflects a real person’s voice, get explicit consent for cloning/imitation.
- **Attribution:** document dataset provenance and license terms clearly.
- **Misuse mitigation:** add a usage policy if the model can be used for impersonation.

---

## 9) Evidence Appendix (from `TTS_Model.ipynb`)

## 10) Contact / Maintainer

- Maintainer: Nunsi
- Dataset used in training notebook: `Nunsi/tts-data`
- Base model: `unsloth/csm-1b`
