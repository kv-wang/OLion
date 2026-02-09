<div align="center">

# OLion

**OLion**: Approaching the Hadamard Ideal by Intersecting Spectral and ℓ∞ Implicit Biases


<div>

</div>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://arxiv.org/abs/2505.12284" target="_blank">
    <img alt="Arxiv"
    src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white"/></a>
</div>

<div>
<br>

</div>

## News
[2026/1/29] We release our paper [OLion: Approaching the Hadamard Ideal by Intersecting Spectral and ℓ∞ Implicit Biases](https://www.arxiv.org/abs/2602.01105) and update the main branch



## Overview




We introduce **OLion**(Orthogonal Lion), an efficient and effective optimizer which
combines spectral control from orthogonalized update directions with ℓ∞-style coordinate control
from sign updates.



## Getting Started 🚀

### Installation & Training Scripts

#### nanoGPT Setup
To begin working with **Short-RL** for the Logic-RL dataset, just run:

```bash
cd nanoGPT
conda create -n nanogpt python=3.10
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
#### Llama Setup

To begin working with **Llama_2_7b pretraining**, run :

```bash
cd Llama
conda env create -f environment.yml
pip install -r requirements.txt
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
#### SIT pretraining Setup

Please refer to the [REPA](https://github.com/sihyun-yu/REPA) repository.



#### Start nanoGPT Training

We directly use the openwebtext dataset

Train GPT2 small with OLion optimizer:

```bash
cd nanoGPT
bash run.sh 
```
You can modify the optimizer, batch size, learning rate and model scale in run.sh

#### Start Llama 2 7b Training
To run the llama 2 7b pretraining experiment with OLion:
```bash
cd Llama
bash run_llama_2_yb.sh
```
You can modify the training configurations in Llama\train_configs\llama2_7b.toml, where you can chose the optimizer, and change 
other configs including learning rate, batch size and dataset according to your need.


#### Start SiT-B/2 Training
To run SiT-B/2 pretraining with OLion:
```bash
cd SIT
bash run.sh
```
You can modify the training configurations in SIT/run.sh .


## Acknowledgements

Our training framework is built on [nanoGPT](https://github.com/karpathy/nanoGPT), [torchtitan](https://github.com/pytorch/torchtitan), and [REPA](https://github.com/sihyun-yu/REPA).


## Citation

```bibtex
@misc{wang2026olionapproachinghadamardideal,
      title={OLion: Approaching the Hadamard Ideal by Intersecting Spectral and $\ell_{\infty}$ Implicit Biases}, 
      author={Zixiao Wang and Yifei Shen and Huishuai Zhang},
      year={2026},
      eprint={2602.01105},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01105}, 
}
```