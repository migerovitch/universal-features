# Sparse Autoencoder Universality: Under What Conditions are Learned Features Consistent?

## Introduction

Neural networks are black boxes. We understand the process by which they are created, but just as understanding the principle of evolution yields little insight into the human brain, designing a model’s optimization process yields little insight into how that model reasons. The field of mechanistic interpretability attempts to understand how human-understandable concepts combine within a model to form its output. With sufficiently good interpretability tools, we could ensure reasoning transparency and easily find and remove harmful capabilities within models. 

In 2022, Anthropic identified a core challenge in interpreting a model’s reasoning layer-by-layer: polysemanticity, a phenomenon in which a single neuron activates for many different concepts (e.g. academic citations, English dialogue, HTTP requests, and Korean text). This is a result of a high-dimensional space of concepts (‘features’) being compressed into the lower-dimension space of the neural network (https://arxiv.org/abs/2209.10652). Sparse autoencoders, a form of dictionary learning, help to linearly disentangle polysemantic neurons into interpretable features (https://arxiv.org/abs/2309.08600) .  

Sparse autoencoders work by projecting a single layer of a neural network into a higher-dimension space (in our experiments, we train autoencoders ranging from a 1:1 projection to a 1:32 projection) and then back down to the size of the original layer. They are trained on a combination of reconstruction loss, their ability to reconstruct the original input layer, and a sparsity penalty, encouraging as many weights as possible to be 0 while retaining good performance (https://arxiv.org/abs/2309.08600). 

## About this project

Asher Parker-Sartori and Misha Gerovitch completed this project as our final for 6.S898 Deep Learning at MIT. This is a work in progress and we are working to expand on our results in the future.

## About the codebase

- easy_training_base.(py/ipynb) as well as some other helper code was copied (with adaptations) from [loganriggs/sparse_coding](https://github.com/loganriggs/sparse_coding) to train our own sparse autoencoders.
- analysis and data_visualization directories contain the scripts we used to create the visuals and results sections for our project.
- we started off by analysing pre-trained sparse autoencoders from [samrmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning).
