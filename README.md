# Harmonic Loss, optimized implementation

Optimized **Harmonic Loss** for classification with two key compute tricks. Ready for large models and NLP or vision training.

## What it does

*   Builds **harmonic logits** from **Euclidean distances** between the hidden state and per-class vectors. Replaces the standard dot-product view with a class-centers geometry.
*   Avoids massive intermediates by using the **algebraic expansion** of the squared distance. Memory goes from O(B·S·V·C) to O(B·S·V).
*   Works in **log space** for numerical stability and is **exactly equivalent** to HarMax with distances when you set the exponent correctly.

## Core math

*   Harmonic probabilities for class $i$:

    $$
    p_i=\frac{d_i^{-n}}{\sum_j d_j^{-n}},\quad d_i=\lVert w_i-x\rVert_2.
    $$

    Use **harmonic logits** $z_i = -n\log d_i$. Softmax over $z$ recovers the same $p$.

*   **Algebraic expansion** without building huge tensors:

    $$
    \lVert w_i-x\rVert^2=\lVert w_i\rVert^2+\lVert x\rVert^2-2\langle w_i,x\rangle.
    $$

    Compute with one GEMM, two norms, and a broadcast.

*   **Gradients in log space** match the direct formulation: $\frac{\partial L}{\partial z_i} = p_i - \mathbf{1}_{i=c}$ and $\frac{\partial z_i}{\partial d_i} = -\frac{n}{d_i}$.

## Important note about the exponent

If you compute $\tilde d_i = \lVert w_i - x \rVert^2$ and then set $z_i = -n\log\tilde d_i$, you get

$$
z_i = -n\log(d_i^2) = -(2n)\log d_i,
$$

which **doubles the effective exponent**. To match the paper, do **one** of these:

1.  Keep squared distances and correct in the log:

    $$
    z_i=-\frac{1}{2}\,n\,\log \tilde d_i.
    $$

2.  Or take the square root before the log and keep $z_i = -n\log d_i$.

The paper suggests $n \approx \sqrt{N}$ for stability. If you stay with $\log(d^2)$, apply the factor $1/2$ in the log or halve $n$.

## Module interface

*   **Constructor**
    `HarmonicLoss(hidden_size, num_classes, harmonic_n=None, eps=1e-6)`

    *   `harmonic_n`: if not given, the module sets $n=\lceil\sqrt{N}\rceil$. Remember the $1/2$ factor if you log squared distances.
    *   `eps`: minimum clamp to avoid $\log 0$.

*   **Forward**
    Inputs:

    *   `hidden_features`: $[B, N]$
    *   `weight_matrix`: $[C, N]$
    *   `targets`: $[B]$
        Outputs:
    *   `loss` scalar and `harmonic_logits` $[B, C]$
        The loss uses cross entropy over the harmonic logits, which matches the harmonic formulation.

## Complexity and memory

*   Dominant cost: **GEMM** $X W^\top$, same as standard softmax classifiers.
*   Extra cost: norms and logs, with **memory footprint** close to cross entropy.

## Useful theoretical properties

*   **Finite-norm convergence** toward class centers under training.
*   **Scale invariance**: scaling all distances by $\alpha$ adds a constant to all logits. Softmax removes it, so $p$ is unchanged.

## Practical tips

*   Set $n\approx \sqrt{N}$. If you use $\log(d^2)$, multiply the log by $1/2$ or divide $n$ by 2.
*   Clamp with a small `eps` before the log for stability.
*   Cache $\lVert W\rVert^2$ if class vectors are static to save compute.

## Internal computation steps

1.  **GEMM**: $y=XW^\top$.
2.  **Norms**: $\lVert X\rVert^2$ per position and $\lVert W\rVert^2$ per class.
3.  **Reconstruction**: $d^2=\lVert X\rVert^2+\lVert W\rVert^2-2y$.
4.  **Clamp + log**: set harmonic logits as $z=-n\log d$ with the exponent fix described above.

## References

*   Baek, Liu, Tyagi, Tegmark. *Harmonic Loss Trains Interpretable AI Models*. arXiv:2502.01628v2, 2025.
*   *Harmonic Loss: Computational Optimizations for Scalable Implementation*.
