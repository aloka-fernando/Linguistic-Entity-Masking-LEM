## Linguistic Entity Masking(LEM)

### Overview
This repository releases the source for Linguistic Entity Masking (LEM), which is aimed at improving the cross-lingual representations of an already pre-trained multilingual embedding model.
We have evaluated the success of LEM strategy for low-resource language pairs Sinhala-Tamil, Sinhala-English and Tamil-English language pairs. The LEM improvement is applied as a continual pre-training step on top of XLM-R multilingual pre-trained language model, with monolingual data and parallel data respectively. 

Separate scripts are available for LEM based continual pre-training with monolingual data (LEM with MLM) and parallel data (LEM with TLM).

If you use this work in your research please cite the following paper

@article{fernando2025linguistic,
  title={Linguistic Entity Masking to Improve Cross-Lingual Representation of Multilingual Language Models for Low-Resource Languages},
  author={Fernando, Aloka and Ranathunga, Surangika},
  journal={arXiv preprint arXiv:2501.05700},
  year={2025}
}
