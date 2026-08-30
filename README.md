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
- [x] A small neural net (MLP) built on top of `Scalar`
- [x] Demo the neural net on the XOR dataset
- [x] Update README with description of challenges faced and how I overcame them

## Challenges

A bug that took a while to track down was applying ReLU to the last layer output. This bug essentially squashed negative outputs from the last layer to zero, which breaks the loss function that I used, since the output of the network is supposed to be either positive or negative. This bug was fixed after correcting an off by one error when checking for the index of the last layer.

---
Inspired by Andrej Karpathy's micrograd