# Satellite Image Captioning on VRSBench

A complete experimental study of **remote-sensing image captioning on
VRSBench**, comparing encoder-decoder and Vision-Language Model (VLM)
approaches with parameter-efficient fine-tuning, decoding ablations, and
length-aware optimization.

> **Task:** Image Captioning on VRSBench\
> **Track:** Internal Hackathon --- Multi-Model Selection\
> **Submission:** Anonymous

------------------------------------------------------------------------

## 1. Project Overview

This project investigates how different vision-language architectures,
fine-tuning strategies, decoding methods, and caption-length control
techniques affect image-captioning performance on remote-sensing
imagery.

The project compares:

-   **BLIP-base** as an encoder-decoder baseline.
-   **Moondream2** and **PaliGemma-3B** as captioning-oriented VLMs.
-   **SmolVLM2-2.2B**, **Qwen3-VL-2B**, **Qwen2.5-VL-7B**, and
    **Qwen3-VL-4B** as general-purpose VLMs.
-   **QLoRA/LoRA fine-tuning** for efficient adaptation.
-   Multiple decoding strategies, including greedy decoding and beam
    search.
-   An auxiliary **length-aware loss** designed around the BERT-BLEU4
    evaluation metric.

The final best configuration in the reported experiments is
**Qwen2.5-VL-7B with domain fine-tuning and greedy decoding with
repetition penalty**, reaching approximately **0.685 BERT-BLEU4** on the
evaluated official validation subset.

------------------------------------------------------------------------

## 2. Dataset

The project uses the **VRSBench captioning dataset**.

### Dataset statistics

  Property                        Value
  ------------------------ ------------
  Total captioned images         20,264
  Image resolution            512 × 512
  Mean caption length        53.0 words
  Median caption length        52 words
  Minimum caption length        7 words
  Maximum caption length      146 words
  Vocabulary size                 4,499
  Missing captions                    0
  Duplicate captions                  0

The training split is used for fine-tuning, while the official
evaluation images are kept separate from training.

### Why VRSBench is challenging

Remote-sensing captions differ from conventional image-captioning
datasets because they often contain:

-   Fine-grained spatial descriptions.
-   Positional phrases such as *top-left*, *bottom-right*, and *adjacent
    to*.
-   Detailed object relationships.
-   Long descriptive captions.
-   Ambiguous aerial scenes where visually similar categories can cause
    hallucination.

------------------------------------------------------------------------

## 3. Project Pipeline

``` text
VRSBench Dataset
       │
       ▼
Data Cleaning + EDA
       │
       ├───────────────┐
       ▼               ▼
   BLIP-base        VLM Models
       │               │
       │          QLoRA / LoRA
       │               │
       ▼               ▼
 Fine-tuning       Domain SFT
       │               │
       └───────┬───────┘
               ▼
       Decoding Strategies
       │
       ├── Greedy
       ├── Beam Search
       └── Repetition Penalty
               │
               ▼
        Caption Generation
               │
               ▼
        BERT-BLEU4 Evaluation
               │
               ▼
     Qualitative + Quantitative
             Analysis
```

------------------------------------------------------------------------

## 4. Evaluation Metric

The primary metric is **BERT-BLEU4**.

The implementation uses phrase-level embeddings from `all-MiniLM-L6-v2`.
The metric combines semantic n-gram similarity with a length penalty.

Conceptually:

``` text
BERT-BLEU4
    =
Semantic n-gram similarity
    ×
Length Penalty
```

The length penalty is symmetric, meaning both overly short and overly
long captions can be penalized.

This makes caption length particularly important for VRSBench.

------------------------------------------------------------------------

## 5. Length-Aware Training

A major part of the project is explicitly accounting for caption length
during training.

### Problem

Standard cross-entropy optimizes token prediction but does not directly
penalize a model for systematically generating captions that are too
short or too long.

This matters for VRSBench because the evaluation metric contains an
explicit length penalty.

### Auxiliary length loss

The final objective is:

``` text
L = L_CE + λ × ((l - l*)² / l*²)
```

where:

-   `L_CE` = standard language-model cross-entropy loss.
-   `l` = generated response length.
-   `l*` = target caption length.
-   `λ = 0.02`.
-   Target length ≈ **53 words**.
-   A hard **90-word ceiling** is applied.

### Why this formulation?

The squared relative error makes both cases costly:

``` text
Too short  ───────► penalty
Target ≈ 53 words ─► minimum penalty
Too long   ───────► penalty
```

The objective does **not** force every caption to contain exactly 53
words. Instead, it encourages the generated-caption distribution to stay
close to the natural VRSBench distribution while allowing the
language-model loss to determine the actual semantic content.

This is especially important because a caption can be semantically
correct but still lose metric score because of a substantial length
mismatch.

------------------------------------------------------------------------

## 6. BLIP-base Experiments

BLIP-base was used as the encoder-decoder baseline.

### Main improvements explored

1.  **Encoder adaptation**
    -   Fully frozen encoder.
    -   Last two vision blocks unfrozen.
2.  **Optimization**
    -   Flat learning rate.
    -   Differential encoder/decoder learning rates.
    -   Optuna-based learning-rate tuning.
    -   Cosine schedule with warm-up.
3.  **Data augmentation**
    -   Geometric flips were removed because horizontal flipping can
        invalidate spatial language.
    -   Photometric augmentation was retained.
4.  **Decoding**
    -   Greedy decoding.
    -   Beam search.
    -   Repetition penalty.
    -   No-repeat n-gram constraint.
    -   Length constraints.
5.  **Tokenization**
    -   Dynamic-length padding.
    -   95th-percentile target length rather than an overly restrictive
        fixed token limit.

### BLIP results

  Configuration                   BERT-BLEU4
  ----------------------------- ------------
  Frozen encoder + greedy              0.519
  Beam + length control                0.565
  Differential LR + Optuna         **0.575**
  Full-data extended training          0.563

The best BLIP configuration achieved **0.575 BERT-BLEU4**.

------------------------------------------------------------------------

## 7. VLM Fine-Tuning

The VLM experiments use **4-bit QLoRA** to reduce the number of
trainable parameters and memory requirements.

### LoRA configuration

  Parameter                       Value
  ----------------------- -------------
  Quantization                    4-bit
  LoRA rank                          16
  LoRA alpha                         16
  LoRA dropout                     0.05
  Optimizer                 8-bit AdamW
  Learning rate                2 × 10⁻⁴
  Scheduler                      Cosine
  Warm-up                     100 steps
  Per-device batch size               2
  Gradient accumulation               8
  Effective batch size               16

LoRA adapters are applied to language-backbone attention and MLP
projections while most pretrained parameters remain frozen.

------------------------------------------------------------------------

## 8. Model Comparison

The main reported results are:

  Model / Configuration                               BERT-BLEU4
  ------------------------------------------------- ------------
  BLIP-base                                                0.575
  Moondream2                                               0.581
  PaliGemma-3B                                             0.568
  SmolVLM2-2.2B + length loss                              0.511
  Qwen3-VL-2B                                              0.478
  Qwen2.5-VL-7B initial/SFT configuration                  0.436
  Qwen3-VL-4B initial                                      0.430
  Qwen2.5-VL-7B + beam search                              0.671
  **Qwen2.5-VL-7B + greedy + repetition penalty**      **0.685**

The results show that parameter count alone does not determine
captioning performance. Fine-tuning, architecture type, decoding
strategy, and output-length behavior all have substantial effects.

------------------------------------------------------------------------

## 9. Decoding Ablation

For the fine-tuned Qwen2.5-VL-7B model:

  Strategy                             BERT-BLEU4   Length Penalty   Predicted Length
  ---------------------------------- ------------ ---------------- ------------------
  Beam search, k=3                          0.671            0.844               53.3
  Greedy + repetition penalty 1.05      **0.685**        **0.853**               58.8

The greedy configuration produced the strongest reported score.

This indicates that for this task, more complex beam-level search does
not necessarily produce the best metric result. The combination of
semantic quality and controlled repetition was more effective.

------------------------------------------------------------------------

## 10. Parameter Count vs. Performance

One of the project analyses compares model size with BERT-BLEU4.

The main observation is:

> **Increasing parameter count does not automatically produce better
> captioning performance.**

For example, smaller captioning-oriented VLMs can outperform larger
general-purpose VLM configurations when the latter are not sufficiently
adapted to the remote-sensing domain.

The analysis therefore treats **architecture and training strategy** as
important factors alongside model scale.

------------------------------------------------------------------------

## 11. Caption Length Analysis

The project also analyzes the relationship between average generated
caption length and BERT-BLEU4.

The VRSBench caption distribution is centered around approximately **53
words**, while several general-purpose chat VLMs tend to generate
substantially longer captions.

The analysis shows that:

-   Very short captions can lose semantic coverage.
-   Very long captions can incur the metric's length penalty.
-   Captioning-native models tend to stay closer to the dataset's
    natural length range.
-   The best Qwen configuration can still perform strongly with moderate
    length overshoot because its semantic alignment is stronger.

This supports the use of explicit length-aware training and metric-aware
prompting.

------------------------------------------------------------------------

## 12. Prompting Strategy

For applicable VLMs, the prompting strategy was made metric-aware.

The prompt encourages:

-   Approximately **40--55 words**.
-   Direct image description.
-   No unnecessary hedging.
-   No meta-commentary.
-   No explanation of the generation process.
-   Consistent captioning style.

Held-out examples can also be used as in-context anchors where supported
by the architecture.

------------------------------------------------------------------------

## 13. Qualitative Analysis

Quantitative metrics are complemented by visual inspection of generated
captions.

A representative example demonstrates that the model can correctly
identify:

-   The overall barren/arid landscape.
-   Terrain structure.
-   Visible tracks.
-   The presence of a wind turbine/windmill.

However, the model may still make **spatial localization errors**, such
as placing an object in the wrong quadrant.

This highlights a key limitation of current VLM captioning systems:
strong semantic recognition does not guarantee precise spatial
grounding.

------------------------------------------------------------------------

## 14. Observed Failure Modes

### 1. Spatial hallucination

The model may identify the correct object but assign it an incorrect
location.

### 2. Object-count errors

Models may generate an incorrect number of objects.

### 3. Scene-type confusion

Visually similar remote-sensing scenes can be mapped to the wrong
semantic category.

### 4. Over-generation

General-purpose chat VLMs may generate unnecessarily long captions.

### 5. Hedging

Instruction-tuned models can sometimes add unnecessary phrases or
meta-commentary that are not useful for benchmark captioning.

------------------------------------------------------------------------

## 15. Geometric Augmentation Finding

Horizontal flips were intentionally removed from the final augmentation
pipeline.

Remote-sensing captions frequently contain positional information such
as:

``` text
top-left
bottom-right
left of
right of
adjacent to
```

A horizontal flip changes the spatial interpretation of these
statements.

Therefore, geometric transformations can introduce a mismatch between
the transformed image and its original caption.

Photometric augmentation such as brightness/contrast variation was
retained.

------------------------------------------------------------------------

## 16. Training Analysis

The BLIP training logs show rapid reduction in training loss during the
early epochs followed by slower convergence.

Representative validation measurements include:

    Epoch   Train Loss   BERT-BLEU4
  ------- ------------ ------------
        2       1.1484       0.5582
        4       0.9826       0.5600

The broader experiments indicate that simply continuing training does
not guarantee better validation performance. This motivated comparison
of different training schedules and configurations rather than relying
only on training-loss reduction.

------------------------------------------------------------------------

## 17. Reproducibility

### Recommended environment

-   Python 3.10+
-   PyTorch
-   Transformers
-   PEFT
-   Accelerate
-   BitsAndBytes
-   Sentence Transformers
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   Jupyter / Kaggle / Colab

Example environment setup:

``` bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install transformers peft accelerate bitsandbytes
pip install sentence-transformers
pip install numpy pandas matplotlib scikit-learn
pip install jupyter
```

For GPU environments, install the PyTorch build appropriate for the
available CUDA version.

------------------------------------------------------------------------

## 18. Suggested Project Structure

``` text
VRSBench-Image-Captioning/
│
├── README.md
│
├── data/
│   ├── train/
│   ├── Images_val/
│   └── VRSBench_EVAL_Cap.json
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_blip_training.ipynb
│   ├── 03_vlm_baselines.ipynb
│   ├── 04_qwen_qlora_training.ipynb
│   ├── 05_decoding_ablation.ipynb
│   └── 06_evaluation_analysis.ipynb
│
├── src/
│   ├── data.py
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│   └── utils.py
│
├── checkpoints/
│   └── ...
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── logs/
│
├── figures/
│   ├── caption_length_distribution.png
│   ├── blip_training_curve.png
│   ├── model_vs_bert_bleu4.png
│   ├── params_vs_bert_bleu4.png
│   ├── length_vs_bert_bleu4.png
│   └── qualitative_example_clean.png
│
└── report/
    ├── VRSBench_Updated_Report.tex
    └── VRSBench_Updated_Report.pdf
```

------------------------------------------------------------------------

## 19. Running the Project

### Step 1 --- Prepare the dataset

Place the VRSBench training images and caption annotations in the
expected dataset directories.

Verify:

``` text
20,264 caption-image pairs
512 × 512 image resolution
No missing captions
```

### Step 2 --- Run EDA

Analyze:

-   Caption-length distribution.
-   Mean/median/max caption length.
-   Vocabulary size.
-   Duplicate/missing captions.
-   Image dimensions.

### Step 3 --- Train BLIP

Start with the frozen encoder baseline and progressively test:

``` text
Frozen encoder
      ↓
Unfreeze last 2 vision blocks
      ↓
Differential learning rates
      ↓
Improved decoding
```

### Step 4 --- Fine-tune VLMs

Use QLoRA/LoRA for parameter-efficient adaptation.

### Step 5 --- Evaluate

Generate captions on the official evaluation subset and calculate:

``` text
BLEU-1
BLEU-2
BLEU-3
BLEU-4
METEOR
ROUGE-L
CIDEr
BERT-BLEU4
Average caption length
```

### Step 6 --- Compare decoding strategies

Evaluate at least:

``` text
Greedy decoding
Beam search
Greedy + repetition penalty
```

### Step 7 --- Perform qualitative analysis

Inspect representative examples for:

-   Object recognition.
-   Spatial grounding.
-   Caption completeness.
-   Hallucination.
-   Length behavior.

------------------------------------------------------------------------

## 20. Main Findings

### Finding 1 --- Length matters

The symmetric BERT-BLEU4 length penalty makes caption length a
first-order optimization factor.

### Finding 2 --- Model size is not enough

A larger VLM does not automatically outperform a smaller model.

### Finding 3 --- Domain adaptation is important

Fine-tuning on VRSBench substantially improves the behavior of
general-purpose VLMs.

### Finding 4 --- Decoding can change the final score

Greedy decoding with a mild repetition penalty produced the best
reported Qwen2.5-VL-7B result.

### Finding 5 --- Spatial hallucination remains difficult

Models can correctly identify a scene while still making object-count,
category, or spatial-position errors.

### Finding 6 --- Geometric augmentation can be harmful

Spatial transformations can invalidate positional language contained in
the reference captions.

------------------------------------------------------------------------

## 21. Best Reported Configuration

``` text
Model
└── Qwen2.5-VL-7B

Fine-tuning
└── 4-bit QLoRA / LoRA

Decoding
└── Greedy

Repetition penalty
└── 1.05

Evaluation
└── BERT-BLEU4

Best reported score
└── 0.6851
```

------------------------------------------------------------------------

## 22. Report and Figures

The project includes an IEEE-style technical report covering:

-   Dataset analysis.
-   Evaluation metric.
-   BLIP architecture and experiments.
-   VLM fine-tuning.
-   Length-aware training.
-   Decoding ablation.
-   Cross-model comparison.
-   Parameter-performance analysis.
-   Caption-length analysis.
-   Qualitative examples.
-   Failure modes and conclusions.

Generated figures include:

-   Caption-length histogram.
-   BLIP training curve.
-   Model vs. BERT-BLEU4.
-   Parameter count vs. BERT-BLEU4.
-   Caption length vs. BERT-BLEU4.
-   Qualitative captioning examples.

------------------------------------------------------------------------

## 23. Limitations

The experiments are subject to several limitations:

-   Some VLM comparisons use a smaller official validation subset than
    the BLIP experiments.
-   BERT-BLEU4 results depend on the embedding model used for
    evaluation.
-   Qualitative hallucination analysis is based on representative
    examples rather than exhaustive error annotation.
-   The best Qwen result is reported on the evaluated 50-image official
    validation subset.
-   Spatial grounding remains an unresolved limitation of pure
    captioning architectures.

------------------------------------------------------------------------

## 24. Conclusion

This project presents a systematic study of remote-sensing image
captioning on VRSBench. The experiments show that strong performance
depends not only on model scale, but also on **domain-specific
fine-tuning, caption-length behavior, prompting, and decoding
strategy**.

The best reported configuration combines **Qwen2.5-VL-7B, QLoRA-based
domain adaptation, metric-aware generation, and greedy decoding with
repetition control**, achieving approximately **0.685 BERT-BLEU4**.

The experiments also show that future improvements should focus on
**spatial grounding, hallucination reduction, and better metric-aware
multimodal training**, rather than simply increasing model size.

------------------------------------------------------------------------

## 25. Citation

If using VRSBench, cite the original benchmark paper:

``` bibtex
@article{zheng2024vrsbench,
  title={VRSBench: A Comprehensive Vision-Language Benchmark for Remote Sensing},
  author={Zheng, Z. and Chen, X. and Zou, Z. and Shi, Y. and Jiang, H. and Li, J. and Bai, X. and Shen, W. and Zhang, L.},
  year={2024},
  journal={arXiv preprint arXiv:2403.20187}
}
```

For BLIP:

``` bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

For LoRA:

``` bibtex
@article{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

------------------------------------------------------------------------

## 26. Acknowledgement of Experimental Scope

All numerical results in this README correspond to the experiments
documented in the accompanying project report. Training configurations,
validation subsets, evaluation settings, and model variants should be
kept consistent when reproducing the reported numbers.
