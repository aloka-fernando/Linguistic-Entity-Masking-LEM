## Linguistic Entity Masking (LEM)

### Overview

This repository releases the source code for **Linguistic Entity Masking (LEM)**, a strategy designed to improve the cross-lingual representations of already pre-trained multilingual embedding models.

We evaluate the effectiveness of the LEM strategy on **low-resource language pairs**, specifically Sinhala-Tamil, Sinhala-English, and Tamil-English. The LEM technique is applied as a **continual pre-training step** on top of the XLM-R multilingual pre-trained language model, using both **monolingual** and **parallel** corpora.

Separate scripts are provided for:

- **LEM with MLM**: Continual pre-training using *monolingual data* with a masked language modeling objective.
- **LEM with TLM**: Continual pre-training using *parallel data* with a translation language modeling objective.

### 📖 Citation

If you use this work in your research, please cite the following paper:

```bibtex
@article{fernando2025linguistic,
  title={Linguistic entity masking to improve cross-lingual representation of multilingual language models for low-resource languages: A. Fernando, S. Ranathunga},
  author={Fernando, Aloka and Ranathunga, Surangika},
  journal={Knowledge and Information Systems},
  volume={67},
  number={11},
  pages={9905--9946},
  year={2025},
  publisher={Springer}
}
