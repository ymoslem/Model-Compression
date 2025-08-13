# Model-Compression





## Citation

```bib
@inproceedings{moslem-2025-efficient,
    title = "Efficient Speech Translation through Model Compression and Knowledge Distillation",
    author = "Moslem, Yasmin",
    booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.iwslt-1.40/",
    doi = "10.18653/v1/2025.iwslt-1.40",
    pages = "379--388",
    ISBN = "979-8-89176-272-5",
    abstract = "Efficient deployment of large audio-language models for speech translation remains challenging due to their significant computational requirements. In this paper, we address this challenge through our system submissions to the `Model Compression' track at the International Conference on Spoken Language Translation (IWSLT 2025). We experiment with a combination of approaches including iterative layer pruning based on layer importance evaluation, low-rank adaptation with 4-bit quantization (QLoRA), and knowledge distillation. In our experiments, we use Qwen2-Audio-7B-Instruct for speech translation into German and Chinese. Our pruned (student) models achieve up to a 50{\%} reduction in both model parameters and storage footprint, while retaining 97-100{\%} of the translation quality of the in-domain (teacher) models."
}
```
