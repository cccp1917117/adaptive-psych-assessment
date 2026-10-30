# VAE-Based Questionnaire Shortening — Experimental Scaffold

**Status: working synthetic-data prototype; real-scale research is planned.**

This extension explores whether a compact set of answered items can support reconstruction of a longer response vector. It is separate from the repository's existing FastAPI/Streamlit, rule-based follow-up system. It does not yet change that application.

**No real participant data, copyrighted questionnaire text, clinical thresholds, or validated short form are included.** The initial scaffold was generated with AI assistance and requires project-owner review and further experimentation before research claims are made.

## Implemented

- Artificial 0–4 item responses from a three-factor simulator.
- Participant-disjoint train / selection-validation / test splits (60/20/20).
- A masked-input VAE with observed-value/mask inputs and categorical item outputs.
- Greedy selection of one fixed short-form item subset using validation data only.
- Held-out response reconstruction and full mean-score error diagnostics.
- Random item selection and training-set item-mean imputation baselines.
- Eight automated checks, including a hidden-answer leakage check.

## Quick start

Python 3.10 or newer. Run these commands from the repository root:

```bash
cd experiments/vae_short_form
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py --n 600 --items 18 --short 6 --epochs 20 --seed 42
```

The example saves `outputs/synthetic_demo.json` with the configuration, split sizes, selected zero-based item indices, training history, test metrics, and software versions. Its status explicitly says **SYNTHETIC DEMO ONLY**. The CPU implementation was smoke-tested end-to-end; this is not a benchmark claim. Output and model artifacts are ignored by Git.

## Method

```text
Artificial full responses
  -> split participants before training
  -> mask some answers; encode [observed values, observed mask]
  -> Gaussian latent variable -> categorical decoder
  -> select K items on validation participants
  -> freeze item subset; evaluate on untouched test participants
```

A low-dimensional latent representation alone does **not** reduce the number of questions someone answers. The separate item-selection step is what defines the candidate short form.

For each training participant, the model receives only the randomly observed answers. The objective is categorical reconstruction cross-entropy (averaged across items) plus a beta-weighted Gaussian KL term (summed across latent dimensions, averaged across participants). This is a masked-denoising VAE-style objective. The baseline treats answer categories categorically rather than imposing an ordinal likelihood.

At inference, decoding uses the encoder's posterior mean, not Monte Carlo posterior predictive integration. The expected category value is used to reconstruct unadministered responses; administered responses are preserved exactly. A distinct observation mask prevents an unanswered item from being confused with a genuine score of zero.

Greedy selection minimizes validation full-response MSE, including the known answers. This is a simple heuristic, not an optimal subset guarantee. Item selection, model tuning, or stopping decisions must not use test participants.

## Reading the results

`vae_greedy` and `vae_random` use the same trained model but different fixed item subsets. `train_mean_greedy` and `train_mean_random` impute unadministered responses with means calculated from training participants only. Random selection is one seeded baseline, not a multi-seed estimate.

- `unadministered_item_rmse`: reconstruction error on unanswered items only.
- `all_item_rmse_including_observed`: all-item error, which is partly lowered mechanically by retaining observed answers.
- `full_mean_score_rmse`: disagreement with the mean of the full synthetic response vector.

**These are engineering diagnostics, not evidence that a shortened psychological scale is reliable or valid.** Different item subsets also change which unadministered items are evaluated. The simulator is multidimensional; its overall mean is only a diagnostic, not a meaningful psychological construct.

## Not implemented / research roadmap

1. Approved real-data loading, item metadata, reverse scoring, and subscale definitions.
2. Content/domain coverage constraints and ordinal item-response modeling.
3. Multi-seed comparisons, repeated or nested validation, and competitive linear/IRT baselines.
4. Independent evaluation of factor structure, reliability, criterion validity, and measurement invariance.
5. Personalized sequential item selection and a justified stopping rule.
6. Integration with the existing API/UI after validation of the intended use.

Do not call this CAT, an IRT implementation, a validated shortened scale, or a diagnostic instrument. Do not report synthetic results as real-participant findings. Do not repeatedly optimize against the held-out test set.

## Data and privacy

Use only appropriately authorized data. Keep raw participant responses, identifying information, private datasets, and non-redistributable item text out of Git. There is no real-data upload or network endpoint in this experiment. Do not connect it to live participant assessment in its current form.

## References

- Kingma & Welling, *Auto-Encoding Variational Bayes*: https://arxiv.org/abs/1312.6114
- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Boateng et al. (2018), *Best Practices for Developing and Validating Scales for Health, Social, and Behavioral Research*: https://doi.org/10.3389/fpubh.2018.00149

These references motivate the components and validation cautions; none establishes the effectiveness of this particular prototype. No research-performance or clinical claims are made.
