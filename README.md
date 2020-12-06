# Text Classification through Glyph-aware Disentangled Character Embedding and Semantic Sub-character Augmentation
Author: Takumi Aoki, Shunsuke Kitada, Hitoshi Iyatomi

Abstract:
*We propose a new character-based text classification framework for non-alphabetic languages, such as Chinese and Japanese. Our framework consists of a variational character encoder (VCE) and character-level text classifier. The VCE is composed of a β-variational auto-encoder (β-VAE) that learns the proposed glyph-aware disentangled character embedding (GDCE). Since our GDCE provides zero-mean unit-variance character embeddings that are dimensionally independent, it is applicable for our interpretable data augmentation, namely, semantic sub-character augmentation (SSA). In this paper, we evaluated our framework using Japanese text classification tasks at the document- and sentence-level. We confirmed that our GDCE and SSA not only provided embedding interpretability but also improved the classification performance. Our proposal achieved a competitive result to the state-of-the-art model while also providing model interpretability.*

Paper: https://arxiv.org/abs/2011.04184

## Download and Make Datasets
Datasets
- IPA font https://moji.or.jp/ipafont
- Livedoor news corpus http://www.rondhuit.com/download.html#ldcc
```
./script/make_datasets.sh
```

## Glyph-aware Disentangled Character Embedding (GDCE)
```
python train_character_embedding.py \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --train_batch 64 \
    --weight_decay 0
```

## Semantic Sub-character Augmentation (SSA)
```
python train_classification.py \
    --dataset livedoor \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --da ssa \
    --gamma 2.0 \
```

## Tensorboard
```
tensorboard --logdir logs
```

## Citation
```
@inproceedings{aoki-etal-2020-text,
    title = "Text Classification through Glyph-aware Disentangled Character Embedding and Semantic Sub-character Augmentation",
    author = "Aoki, Takumi and Kitada, Shunsuke and Iyatomi, Hitoshi",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.aacl-srw.1",
    pages = "1--7",
}
```