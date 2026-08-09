# notebooks/
Exploratory notebooks only — nothing here is a dependency of `src/`. Conventions:
- Name as `NN_topic.ipynb` (e.g. `01_chunking_strategy_comparison.ipynb`).
- Clear all outputs before committing (enforced by a `nbstripout` pre-commit hook once notebooks land).
- Any result worth keeping gets promoted into `src/evaluation` + `reports/`, not left only in a notebook.
