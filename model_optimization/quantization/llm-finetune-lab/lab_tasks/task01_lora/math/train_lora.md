# LoRA Basic Training Script — Mathematical Formulation (SST-2)

This document provides the mathematical interpretation of every key function
in the `train_lora_sst2.py` script.  
It connects implementation details with their analytical expressions.

---

## 1. `prepare_dataset`

Tokenization converts raw text \( t_i \) into numerical sequences:

$$
x_i = \text{Tokenizer}(t_i) = [\text{CLS}] + \text{VocabIDs}(t_i) + [\text{SEP}]
$$

where each \( x_i \in \mathbb{N}^{L_i} \), padded or truncated to  
a fixed maximum length \( L = L_{\text{max}} = \texttt{config["training"]["max_length"]} \).

The label mapping is:
$$
y_i \in \{0, 1\}, \quad \text{for negative/positive sentiment.}
$$

---

## 2. `apply_lora_to_model`

Each selected linear transformation
$$
h = W x + b
$$
is replaced with a LoRA-augmented version:
$$
h = W x + b + s \cdot B A x
$$

where:
- $A \in \mathbb{R}^{r \times d_{in}}$ (down projection)
- $B \in \mathbb{R}^{d_{out} \times r}$ (up projection)
- $s = \alpha / r$ (scaling factor)

This can equivalently be written as a **merged effective weight**:
$$
W' = W + sBA
$$

---

## 3. `mark_only_lora_as_trainable`

During optimization:
$$
\nabla_\theta \mathcal{L} = 0, \quad \forall \theta \notin \{A, B\}
$$

Only LoRA matrices $A, B$ receive gradient updates;  
the original base weights $W, b$ remain frozen.

---

## 4. `train_one_epoch`

Each training step minimizes the **cross-entropy loss** for classification:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N}
    \sum_{c=1}^{C} y_{ic} \log \hat{p}_{ic}
$$

with predicted logits
$$
z_i = f_\theta(x_i),
\quad
\hat{p}_{ic} = \text{softmax}(z_i)_c
$$

and parameters $\theta = \{A, B\}$.

Backpropagation computes:
$$
\nabla_\theta \mathcal{L} =
\frac{\partial \mathcal{L}}{\partial z_i}
\frac{\partial z_i}{\partial \theta}
$$

Gradients are clipped:
$$
\nabla_\theta \leftarrow
\frac{\nabla_\theta}{\max(1, \|\nabla_\theta\|_2 / g_{\max})}
$$
where $g_{\max} = \text{gradient\_clip}$.

---

## 5. `evaluate_model`

Evaluation computes the same forward pass without gradients:
$$
\hat{y}_i = \arg\max_c \hat{p}_{ic}
$$

Accuracy metric:
$$
\text{Accuracy} =
\frac{1}{N} \sum_{i=1}^{N}
\mathbb{1}\{\hat{y}_i = y_i\}
$$

---

## 6. Optimizer & Scheduler

Optimization uses AdamW:
$$
\theta_{t+1} = \theta_t
- \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}
- \eta_t \lambda \theta_t
$$

with bias-corrected moments:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Learning rate schedule (linear warmup + decay):
$$
\eta_t =
\begin{cases}
\eta_{\max} \cdot \frac{t}{t_{\text{warmup}}}, & t < t_{\text{warmup}} \\
\eta_{\max} \cdot \left(1 - \frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right),
& t \ge t_{\text{warmup}}
\end{cases}
$$

---

## 7. `plot_training_curves`

Given recorded metrics
$$
\{\mathcal{L}_{\text{train}}^{(e)}, \mathcal{L}_{\text{eval}}^{(e)}, 
\text{Acc}_{\text{train}}^{(e)}, \text{Acc}_{\text{eval}}^{(e)}\}_{e=1}^{E},
$$
this function plots:
- $\mathcal{L}$ vs. epoch
- $\text{Accuracy}$ vs. epoch

to visualize convergence trends.

---

## 8. Saving LoRA Weights

The LoRA state dictionary stores only $\{A, B\}$:
$$
\text{LoRA\_state} = \{A, B\}
$$

Full checkpoint includes:
$$
\mathcal{C} = 
\{\text{epoch}, \text{optimizer\_state}, \text{scheduler\_state},
\mathcal{L}, \text{Acc}, \text{config}, \text{LoRA\_state}\}
$$

A minimal adapter file (`lora_adapter.pt`) keeps only:
$$
\text{LoRA\_adapter} = \{A, B, \text{meta config}\}
$$

---

## 9. `merge_lora_weights`

For inference or upload:
$$
W \leftarrow W + sBA
$$

This merges LoRA updates into the frozen base model.

---

## 10. `evaluate / push_to_hub` (optional)

The final merged model $\tilde{W}$ behaves as a standard Transformer:
$$
h = \tilde{W}x + b
$$
with the LoRA correction baked in.

---

## Summary

| Component | Formula |
|------------|----------|
| Forward (LoRA) | $y = (xW^\top + b) + s((xA^\top)B^\top)$ |
| Scaling | $s = \alpha / r$ |
| Merge | $W_{\text{merged}} = W + sBA$ |
| Loss | $\mathcal{L} = -\sum y \log \hat{p}$ |
| Accuracy | $\frac{1}{N}\sum \mathbb{1}\{\hat{y}=y\}$ |
| Optimizer | AdamW with weight decay |
| Scheduler | Linear warmup + decay |
| Gradient clip | $\|\nabla\|_2 \le g_{\max}$ |

---

**Author:** Jiao (LLM Tuning Lab)  
**License:** MIT  
**Dataset:** SST-2 (GLUE Benchmark)  
**Model:** BERT-base with LoRA fine-tuning