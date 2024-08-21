# Plug-in estimation of Schrödinger bridges 

This repository provides a full implementation of our method for estimating Schrödinger bridges on the basis of i.i.d. samples from the source and target measures. Our approach, called the *Sinkhorn bridge*, is based on expressing the time-dependent drift as a function of the static potentials which solve the entropic optimal transport problem on the data. The GIF below is an visualization of our approach for computing the Sinkhorn bridge for three common low-dimensional datasets in the machine learning literature.

<p align="center">
<img align="middle" src="./assets/sinkhornbridge.gif" alt="SINKBRIDGE FIG" width="800" height="200" />
</p>

## Examples
Jupyter notebooks that replicate the experiments found in our paper can be found in [`examples`](./examples). These include the 2D examples in the GIF above, the Gaussian-to-Gaussian setting, and a benchmark setting due to [Grushchin et al. (2023)](https://github.com/ngushchin/EntropicOTBenchmark), where we estimate a non-trivial drift in high-dimensions.   

## Basic usage
Our approach is simulation-free in that we reduce the problem of estimating the drift defining the Schrödinger bridge (in either the forward or backward direction) to estimation of the potentials that define the entropic optimal transport coupling on the data which are computed using Sinkhorn's algorithm [(Cuturi 2013)](https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf). We provide implementations in both [POT](https://pythonot.github.io/) and [OTT-JAX](https://ott-jax.readthedocs.io/en/latest/) frameworks. The method consists of three hyper parameters. First, the user defines the level of noise `eps` which is passed into Sinkhorn's algorithm, and is used to define the drift. Then the user is required to pass the duration of the drift `tau` in [0,1), and the number of steps for the Euler--Maruyama discretization, written `Nsteps`. From here, the estimator takes care of the bridging process which can be initialized at new samples from the source measure. (Note: at`tau = 1` the bridge collapses onto the training data!)

## References

If you found this code helpful, or are building upon this work, please cite 

Aram-Alexandre Pooladian and Jonathan Niles-Weed. "Plug-in estimation of Schrödinger bridges" *arXiv.* 2024. [[arxiv]](https://arxiv.org/abs/TBD)

```
@article{pooladian2024plugin,
  title={Plug-in estimation of {S}chr{\"o}dinger bridges},
  author={Pooladian, Aram-Alexandre, and Niles-Weed, Jonathan},
  journal={arXiv preprint arXiv:TBD},
  year={2024}
}
```
