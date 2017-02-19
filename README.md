# QRNN
Re-implementation of James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. 2016. [Quasi-Recurrent Neural Networks](http://arxiv.org/abs/1611.01576) by Chainer.

The original implementation of QRNN (written in Chainer) is publicly available on this [blog post](https://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/). However, the author only provides so-called "core" implementation, which is not runnable by itself. Also, I believe there is a bug in the implementation of `Linear`, that it does not seem to work in the situation where `in_size != out_size`. So I wrote it by myself.
