# QLoRA Training Script — Mathematical Formulation (Wikitext-2)

### 1) `load_config(config\_path: str) \to \mathrm{Dict}[\mathrm{str}, \mathrm{Any}]$

Parses a YAML file into a configuration dictionary, viewed as a set of hyperparameters:
$$
\mathcal{C}=\{\text{model}, \text{data}, \text{lora}, \text{training}, \text{output}, \text{advanced}\}.
$$

---

### 2) `prepare_dataset(config, tokenizer)`

**Tokenization as a mapping**
For an input text sequence $x$ (string), the tokenizer applies:
$$
T:\ \text{text}\ \mapsto\ \bigl(\texttt{input\_ids},\ \texttt{attention\_mask}\bigr).
$$

**Truncation \& padding to fixed length** $L_{\max}=\texttt{config['data']['max\_length']}$:
$$
\texttt{input\_ids} = \bigl[t_1,\ldots,t_L\bigr]\ \mapsto
\begin{cases}
[t_1,\ldots,t_{L_{\max}}], & L\ge L_{\max} \\
[t_1,\ldots,t_L,\underbrace{t_{\mathrm{PAD}},\ldots,t_{\mathrm{PAD}}}_{L_{\max}-L}], & L<L_{\max}
\end{cases}
$$
$$
\texttt{attention\_mask} = [m_1,\dots,m_{L_{\max}}],\quad
m_i=\begin{cases}
1,& \text{token is real}\\
0,& \text{token is PAD}
\end{cases}
$$

**Causal LM labeling**
For language modeling, labels equal inputs (teacher forcing):
$$
\texttt{labels}=\texttt{input\_ids}.
$$

**Loss at token level** (used later by the trainer):
Given logits $\ell_{t}$ over the vocabulary at position $t$ and gold token $y_t$,
$$
\mathcal{L}_{\mathrm{NLL}} = -\frac{1}{N}\sum_{t\in\mathcal{I}} \log p_\theta(y_t\mid x_{<t})
= -\frac{1}{N}\sum_{t\in\mathcal{I}} \log\frac{e^{\ell_t[y_t]}}{\sum_{v} e^{\ell_t[v]}}.
$$
Here $\mathcal{I}$ indexes non-PAD positions and $N=|\mathcal{I}|$.

---

### 3) `compute\_metrics(eval\_pred)`

The documented metric is **perplexity**:
$$
\mathrm{PPL} = \exp\bigl(\mathcal{L}\bigr),
$$
where $\mathcal{L}$ is the average evaluation loss (cross-entropy) over the validation set.

---

### 4) `plot\_training\_curves(log\_history, save\_path)`

This function visualizes recorded scalars: Training loss $\mathcal{L}_{\mathrm{train}}(s)$, Evaluation loss $\mathcal{L}_{\mathrm{eval}}(s)$, and Learning rate $\eta(s)$ vs. step $s$.

If a **linear warmup** with ratio $\rho$ is used, the schedule is:
$$
\eta(s)=
\begin{cases}
\eta_0\cdot \dfrac{s}{\rho S}, & 0\le s<\rho S,\\[6pt]
\eta_0\cdot f\left(\dfrac{s-\rho S}{S-\rho S}\right), & \rho S\le s\le S,
\end{cases}
$$
where $S$ is total training steps, $\eta_0$ the base LR, and $f(\cdot)$ depends on `lr\_scheduler\_type`.

---

### 5) `main(args)`

This orchestrates the full training. Key math behind the steps:

#### (a) Effective batch size with gradient accumulation

With per-device batch size $B$, gradient accumulation $K$, and $D$ devices:
$$
B_{\text{eff}} = B \times K \times D.
$$

#### (b) Causal language modeling objective

For a tokenized sequence $\mathbf{x}=(x_1,\dots,x_T)$, the model maximizes
$$
\log p_\theta(\mathbf{x})=\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}),
$$
equivalently minimizes average negative log-likelihood (cross-entropy)
$$
\mathcal{L}(\theta)=-\frac{1}{N}\sum_{(i,t)\in\mathcal{D}} \log p_\theta\bigl(x^{(i)}_t\mid x^{(i)}_{<t}\bigr).
$$

#### (c) AdamW (decoupled weight decay) update (conceptual)

For parameter $\theta$, gradient $g_t=\nabla_\theta \mathcal{L}_t$:
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t},
$$
$$
\theta \leftarrow \theta - \eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta\right),
$$
where $\lambda$ is the weight-decay coefficient.

#### (d) Gradient clipping (if enabled by HF defaults)

With clip norm $c$,
$$
g \leftarrow g \cdot \min\left(1,\ \frac{c}{\lVert g\rVert_2}\right).
$$

#### (e) Mixed precision (fp16/bf16)

Forward/backward are computed in the chosen dtype (e.g., BF16). This changes **numerical representation**, not the objective.

#### (f) Evaluation and perplexity

From evaluation loss $\mathcal{L}_{\mathrm{eval}}$ computed by the trainer:
$$
\mathrm{PPL}_{\mathrm{eval}} = \exp\bigl(\mathcal{L}_{\mathrm{eval}}\bigr).
$$

#### (g) Saving adapters

Only LoRA parameters (adapters) $\{A,B\}$ are saved; the effective linear weight during inference is
$$
W' = W + \frac{\alpha}{r}BA,
$$
with $W$ kept frozen/quantized by QLoRA.

---

### Notes on imported utilities used here

* **`load\_model\_and\_tokenizer`** loads a 4-bit NF4–quantized model. Forward uses on-the-fly dequantization $\widehat{W}=\mathrm{Dequantize}(\text{codes}, \widehat{s})$ and computes $y=\widehat{W}x$.
* **`create\_lora\_config`** uses the LoRA low-rank update $\Delta W=\frac{\alpha}{r}BA$.
* **`prepare\_model\_for\_training`** freezes $W$ and trains only $A,B$:
  $$
  \frac{\partial\mathcal{L}}{\partial W}=0,\quad
  \frac{\partial\mathcal{L}}{\partial A}\neq 0,\quad
  \frac{\partial\mathcal{L}}{\partial B}\neq 0.
  $$

---

## Symbol reference

* $x_t$: token at position $t$; $\mathbf{x}$: a token sequence.
* $W$: base linear weight (frozen, quantized); $A,B$: LoRA adapters; $r$: LoRA rank; $\alpha$: LoRA scale.
* $\eta_t$: learning rate at step $t$; $\beta_1,\beta_2,\epsilon$: AdamW hyperparameters; $\lambda$: weight decay.
* $B$: per-device batch size; $K$: gradient accumulation steps; $D$: number of devices; $B_{\mathrm{eff}}$: effective batch size.
* $\mathcal{L}$: average (token-level) cross-entropy loss; $\mathrm{PPL}$: perplexity.
* $S$: total steps; $\rho$: warmup ratio; $f(\cdot)$: scheduler shape (e.g., linear/cosine).

---

**Author:** LLM Tuning Lab
**License:** MIT
**Dataset:** Wikitext-2 (Language Modeling)
**Model:** Qwen2.5-3B (Causal LM) with QLoRA (4-bit NF4) + LoRA adapters