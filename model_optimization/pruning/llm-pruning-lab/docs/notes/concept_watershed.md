Q1. 為何 Unstructured 幾乎不帶來 Latency 降低？
- [Sparse matrix formats（CSR/COO）](https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html)
- [Cache hierarchy / memory access pattern](https://en.wikipedia.org/wiki/Cache_hierarchy)
- Hardware parallelism & kernel scheduling

Q2. Global 稀疏分配何時優於 Layerwise？
- [Fisher Information / Hessian-based sensitivity](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Lee_Masking_Adversarial_Damage_CVPR_2022_supplemental.pdf)
- Lagrange multipliers for constrained pruning optimization
- Saliency-based importance metrics（magnitude, Taylor, movement）

Q3. Network Slimming 中 BN γ 的物理意義是什麼？
- Batch Normalization 數學原理
- L1 regularization 與 sparsity 的關係
- Feature map importance estimation（activation magnitude vs scaling）

Q4. 為何 One-shot 50% 常比 Iterative 10%×5 更難收斂？
- Optimization landscape in deep networks
- Lottery Ticket Hypothesis（Frankle & Carbin, 2019）
- Curriculum pruning / gradual sparsification

Q5. FLOPs 降低 40% 時，實測 Latency 只下降 20% 的可能原因？
- Amdahl’s Law in system optimization
- GPU kernel fusion & operator tiling
- Profiling 指標解讀（FLOPs vs memory bandwidth vs latency）