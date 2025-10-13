# LoRA (Low-Rank Adaptation) — Mathematical Formulation

This document provides the mathematical explanation for each function
in the minimal LoRA implementation compatible with 🤗 Transformers.

---

## Notation and Shapes

| Symbol | Meaning | Shape |
|---------|----------|--------|
| $x$ | Input | $(*, d_{in})$ |
| $W, b$ | Base Linear weights, bias | $(d_{out}\times d_{in}), (d_{out})$ |
| $A, B$ | LoRA down/up projection | $(r\times d_{in}), (d_{out}\times r)$ |
| $s = \alpha / r$ | Scaling factor | scalar |

---

## 1. LoRALayer

**Forward**

$$
x' = \text{Dropout}(x) \\
\Delta y = s \cdot ((x'A^\top)B^\top)
$$

**Reset parameters**

$$
A \sim \text{KaimingUniform}(\cdot), \quad B = 0
$$

---

## 2. LinearWithLoRA

**Forward (unmerged)**  

$$
y = xW^\top + b + s \cdot ((\text{Dropout}(x)A^\top)B^\top)
$$

**Merged weight**  

$$
W_{\text{merged}} = W + sBA
$$

**Unmerge**  

$$
W = W_{\text{merged}} - sBA
$$

---

## 3. apply_lora_to_model

Replace all selected linear layers:

$$
xW^\top + b \;\Rightarrow\; xW^\top + b + s((\text{Dropout}(x)A^\top)B^\top)
$$

---

## 4. Utilities

| Function | Mathematical meaning |
|-----------|----------------------|
| `mark_only_lora_as_trainable` | Only LoRA params have gradients |
| `merge_lora_weights` | $W \leftarrow W + sBA$ |
| `unmerge_lora_weights` | $W \leftarrow W - sBA$ |
| `count_lora_parameters` | count total/trainable params |
| `get_lora_state_dict` | return only $\{A,B\}$ tensors |

---

## Summary

$$
\begin{aligned}
\Delta y &= s((\text{Dropout}(x)A^\top)B^\top) \\
y &= (xW^\top + b) + \Delta y \\
W_{\text{merged}} &= W + sBA
\end{aligned}
$$

---

**Author:** Jiao (LLM Tuning Lab)  
**License:** MIT  