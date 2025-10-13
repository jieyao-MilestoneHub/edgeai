# QLoRA Quantization & LoRA Utilities — Mathematical Formulation


## 1) `create_bnb_config` (4-bit NF4, Double Quantization, BF16)

**Group-wise quantization**

Split weights $w\in\mathbb{R}^n$ into groups of size $g$. For group $t$:
$$
w^{(t)} \in \mathbb{R}^{g}.
$$

NF4 has a fixed non-uniform 4-bit codebook with 16 codepoints:
$$
\mathcal{C}={c_0,c_1,\dots,c_{15}},\quad c_i\in\mathbb{R}.
$$

With a per-group scale $s^{(t)}>0$, quantize each element by nearest code:
$$
k=\arg\min_{i\in{0,\dots,15}}\left|\frac{w^{(t)}_j}{s^{(t)}}-c_i\right|
\quad\Rightarrow\quad
\widehat{w}^{(t)}_j=s^{(t)},c_k.
$$

**Double quantization of scales (8-bit)**
$$
u^{(t)}=\arg\min_{i\in{0,\dots,255}}\left|\frac{s^{(t)}}{s'}-c'*i\right|,
\qquad
\widehat{s}^{(t)}=s'c'*{u^{(t)}},
$$
and final reconstruction:
$$
\widehat{w}^{(t)}_j=\widehat{s}^{(t)},c_k.
$$

**BF16 compute**

$$
\text{compute\_dtype}=\mathrm{bfloat16}.
$$

**Memory (rough)**
$$
\mathrm{Mem}\approx\frac{N,b}{8}\ \text{bytes}\quad(\text{$N$ params, $b$ bits/param, ignoring overheads}).
$$

---

## 2) `create_lora_config` (LoRA hyperparameters)

Given a linear layer $y=Wx$ with $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA constrains the update:
$$
\Delta W=\frac{\alpha}{r},BA,\qquad
A\in\mathbb{R}^{r\times d_{\text{in}}},;B\in\mathbb{R}^{d_{\text{out}}\times r}.
$$

Adapted weight:
$$
W' = W + \Delta W = W + \frac{\alpha}{r}BA.
$$

**LoRA dropout** (training):
$$
\tilde{x}=\mathrm{Dropout}(x;,p=\text{dropout}),\qquad
y = W x + \frac{\alpha}{r}B(A\tilde{x}).
$$

**Trainable params (per linear layer)**
$$
\text{params}*{\text{LoRA}} = r,d*{\text{in}} + d_{\text{out}},r = r(d_{\text{in}}+d_{\text{out}}).
$$

> Apply to attention projections $q_{proj},k_{proj},v_{proj},o_{proj}$ and MLP $gate_{proj},up_{proj},down_{proj}$.

---

## 3) `load_model_and_tokenizer` (loading quantized model & tokenizer)

**On-the-fly dequantization in forward**

\widehat{W}=\mathrm{Dequantize}(\text{NF4_codes},\widehat{s})\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},
\qquad
y=\widehat{W}x.


**Memory comparison (conceptual)**
$$
\mathrm{Mem}*{\mathrm{FP16}}\approx 2N\ \text{bytes},\qquad
\mathrm{Mem}*{4\text{-bit}}\approx 0.5N\ \text{bytes}.
$$

**Right padding**
$$
[t_1,\dots,t_L]\ \mapsto\ [t_1,\dots,t_L,\underbrace{t_{\mathrm{PAD}},\dots,t_{\mathrm{PAD}}}_{L'-L}].
$$

---

## 4) `prepare_model_for_training` (k-bit prep + LoRA + grad checkpointing)

**Freeze base weights; train only LoRA**
$$
\frac{\partial\mathcal{L}}{\partial W}=0,\qquad
\frac{\partial\mathcal{L}}{\partial A}\neq 0,\quad
\frac{\partial\mathcal{L}}{\partial B}\neq 0.
$$

**Gradient checkpointing trade-off**

With $L$ layers split into $K$ segments, activation memory scales roughly as:
$$
\mathrm{Mem}*{\text{acts,no-ckpt}}\propto L\mathcal{A}
\ \Longrightarrow
\mathrm{Mem}*{\text{acts,ckpt}}\propto \frac{L}{K}\mathcal{A},
$$
with recomputation overhead:
$$
\mathrm{Time}*{\text{ckpt}}\approx \mathrm{Time}*{\text{no-ckpt}}\cdot\bigl(1+\rho(K)\bigr),
\quad \rho(K)\in[0.1,0.3]\ \text{(empirical)}.
$$

**Forward with LoRA**
$$
y=\bigl(W+\tfrac{\alpha}{r}BA\bigr)x.
$$

---

以下是您提供的關於記憶體統計和 QLoRA 設定的數學公式，已確認 $\LaTeX$ 格式正確：

## 5) `print_memory_stats` (GPU memory)

$$
\mathrm{Allocated}=\frac{\texttt{torch.cuda.memory\_allocated()}}{10^9},\qquad
\mathrm{Reserved}=\frac{\texttt{torch.cuda.memory\_reserved()}}{10^9},
$$
$$
\mathrm{Free}=
\frac{\texttt{torch.cuda.get\_device\_properties}(0).\texttt{total\_memory}}{10^9}
-\mathrm{Reserved}.
$$

---

## 6) `setup_qlora_model` (one-shot pipeline)

Overall mapping:
$$
(\text{model\_name}; r, \alpha, \text{dropout})
\ \xrightarrow{\ \text{load}\ }\
(\widehat{W},\text{tokenizer})
\ \xrightarrow{\ \text{LoRA}\ }\
\bigl(W'=W+\tfrac{\alpha}{r}BA\bigr)
\ \xrightarrow{\ \text{train}\ }\
\min_{\{A,B\}}\ \mathcal{L}(\theta;\mathcal{D}),
$$
where $\theta$ contains only LoRA params $(A,B)$; base $W$ is quantized \& frozen.

---

### Symbols

$W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ (frozen base weight), $A\in\mathbb{R}^{r\times d_{\text{in}}}$, $B\in\mathbb{R}^{d_{\text{out}}\times r}$, $r$ (rank), $\alpha$ (LoRA scale), $\mathcal{C}$ (NF4 codebook), $s^{(t)}$ & $\widehat{s}^{(t)}$ (group scale & its 8-bit quant), $g$ (group size), $L$ (layers), $K$ (ckpt segments), $\mathcal{A}$ (per-layer activation memory), $\mathcal{L}$ (loss), $\mathcal{D}$ (dataset).

---

**Author:** Jiao (LLM Tuning Lab)  
**License:** MIT  