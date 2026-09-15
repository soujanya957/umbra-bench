# Citations & data credits

umbra-bench includes or derives targets from third-party sources. If you use this
benchmark, cite the sources whose subsets you use.

## MPEG-7 CE-Shape-1 (→ `targets/animals/`)

Standard shape benchmark (1,400 binary silhouettes, 70 classes). We include a small
curated, re-rendered subset; the full dataset is downloaded separately (see
`external/README.md`). Standard citation:

```bibtex
@inproceedings{latecki2000shape,
  title     = {Shape Descriptors for Non-rigid Shapes with a Single Closed Contour},
  author    = {Latecki, Longin Jan and Lak{\"a}mper, Rolf and Eckhardt, Ulrich},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {424--429},
  year      = {2000}
}
```

## HaSPeR (→ `targets/hand_shadow/`)

Real hand-shadow puppet photographs (15,000 images, 15 classes). We keep only a few
binarized exemplar silhouettes per class; the full dataset lives on
[Hugging Face](https://huggingface.co/datasets/Starscream-11813/HaSPeR) and is not
redistributed here (its source clips are used under fair use).

```bibtex
@article{raiyan2024hasper,
  title   = {HaSPeR: An Image Repository for Hand Shadow Puppet Recognition},
  author  = {Raiyan, Syed Rifat and Amio, Zibran Zarif and Ahmed, Sabbir},
  journal = {arXiv preprint arXiv:2408.10360},
  year    = {2024}
}
```

## DejaVu fonts (→ `targets/digits/`, `targets/letters_*/`)

Glyph targets are rendered from the [DejaVu fonts](https://dejavu-fonts.github.io/)
(free license, derived from Bitstream Vera). No citation required; license permits
redistribution of rendered images.

## Related benchmarks (comparison, not included)

```bibtex
@article{xu2025handshadowposer,
  title   = {Hand-Shadow Poser},
  author  = {Xu, Hao and Wang, Yinqiao and Mitra, Niloy J. and Liu, Shuaicheng and Heng, Pheng-Ann and Fu, Chi-Wing},
  journal = {ACM Transactions on Graphics (SIGGRAPH)},
  year    = {2025},
  note    = {210-shape shadow benchmark}
}

@article{mitra2009shadowart,
  title   = {Shadow Art},
  author  = {Mitra, Niloy J. and Pauly, Mark},
  journal = {ACM Transactions on Graphics (SIGGRAPH Asia)},
  volume  = {28},
  number  = {5},
  year    = {2009}
}
```
