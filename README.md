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

When training my XOR net, I noticed that `total_loss` would always plateau and get stuck at `4` after a few epochs. Initially, I thought this was caused by too many neurons accumulating negative weights, so I implemented a feature that counted the number of "dead neurons" (neurons with output 0.0), but noticed that the number of dead neurons in my net was not unusual (2-4 in a net with 11 neurons total).

 It turns out that this was caused by a bug that was applying ReLU to the last layer output. This bug essentially squashed negative outputs from the last layer to zero, which breaks the loss function that I used, since the output of the network is supposed to be either positive or negative. This bug was fixed after correcting an off by one error when checking for the index of the last layer.

---
Inspired by Andrej Karpathy's micrograd