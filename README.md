# sandygrad

An automatic differentiation engine for scalars built from scratch for *educational purposes*.

## What it does

- Builds a computational graph as you write math (a * b + c)
- Computes gradients automatically using `backward()`

## Progress

- [x] Scalar class with `+`, `-`, `*`, `**`, and `ReLU`
- [x] Reverse-mode autodiff via `backward()`
- [x] Support for `/` 
- [x] Unit tests
- [x] A small neural net (MLP) built on top of `Scalar`, trained on a toy dataset

---
Inspired by Andrej Karpathy's micrograd