# external/ — third-party raw data (not committed)

Everything in this directory except this README is gitignored. Raw third-party
datasets are downloaded here, then curated into `targets/` by scripts; only the
small curated derivatives are committed.

## MPEG-7 CE-Shape-1

Download (any of):

- <https://dabi.temple.edu/external/shape/MPEG7/dataset.html> (direct zip)
- <https://biomecis.uta.edu/shape_data.htm>
- [Academic Torrents](https://academictorrents.com/details/0f9ac75f2d9e2ce2ef7b800aa23882915f4e31fa) (needs a BT client, e.g. Transmission)

Unpack so the GIFs sit at `external/MPEG7/<class>-<idx>.gif`, then:

```
python scripts/curate_mpeg7.py
python scripts/build_metadata.py
```

## HaSPeR

Full dataset (15k photos) stays on Hugging Face — do not commit it.

```
pip install -U "huggingface_hub[cli]"
hf download Starscream-11813/HaSPeR --repo-type dataset --local-dir external/HaSPeR
python scripts/curate_hasper.py     # picks ~2 exemplars/class, binarizes
python scripts/build_metadata.py
```

Review `targets/hand_shadow/` visually afterwards — automatic binarization of real
photos is imperfect; delete bad masks and re-pick (PICK_OFFSET in the script).
