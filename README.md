# PISR — Physics-Informed Symbolic Regression

**A Search-Free Foundation Model for Symbolic Regression of Physical Laws**
Chenxi He (Cavendish Laboratory, University of Cambridge) and Jingqiu Chen (Shenzhen University).

PISR recasts symbolic regression as generation within a pre-trained, contrastively-aligned **formula–data
manifold** that carries a **physics prior**: a single forward pass generates candidate formula latents by flow
matching, retrieves others from a real-formula library, decodes and constant-fits them, and returns the best —
no per-instance search. It sets the state of the art on real-physics, cross-domain, and Feynman benchmarks at
matched compute (~2.4× faster than search) and uniquely recovers named laws from raw turbulence, astronomy and
epidemic data.

## Repository contents

| folder | description |
|---|---|
| `code/` | PISR source — shared manifold encoders, flow-matching generator, retrieval, conditioned decoder, training and inference (`train/`, `models/`, `data/`, `configs/`). |
| `scripts/` | Figure-generation scripts. |
| `benchmarks/` | The three evaluation task sets (`merged_physics.jsonl` 797, `clean_nonphys352.jsonl` 270, `feynman.jsonl` 119). |
| `baseline_outputs/` | Per-task top-1 outputs for every method (PISR, PySR, PSE, PhyE2E, GenSR). |
| `case_studies/` | Extraction scripts + access instructions for the turbulence / astronomy / white-dwarf case studies (third-party raw data not redistributed). |
| `figures/` | Paper figures. |

## Model weights and full data (Zenodo)

The pre-trained inference weights (~3.3 GB), the retrieval index, and an archived snapshot of this repository
are on Zenodo: **DOI [10.5281/zenodo.21618861](https://doi.org/10.5281/zenodo.21618861)**. Download the weights
into `weights/` to run inference.

## Quick start

```bash
pip install torch numpy sympy scikit-learn scipy requests
# after placing weights/ from Zenodo:
python -m code.train.eval_dec7 --ckpt weights/manifold_v13_ms80.pt \
    --flow weights/flow_v13.pt --dec weights/dec7_v13.pt \
    --index weights/zf_v13_index.npy --set benchmarks/merged_physics.jsonl
```

## License

Source code under **MIT** (see `LICENSE`); data and model weights under **CC-BY-4.0**. Third-party case-study
data retain their original terms (JHTDB; NASA Exoplanet Archive; Bédard et al. 2020).

## Citation

Please cite the accompanying paper (He & Chen) and the Zenodo record (DOI above).
