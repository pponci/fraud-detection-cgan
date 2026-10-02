## Conventions
- Data: data/<dataset>/{raw,splits,processed}/ (git-ignored)
- Results: results/<dataset>/<experiment>/seed_<n>/ (committed)
- Heavy artifacts: artifacts/<dataset>/<experiment>/seed_<n>/ (git-ignored)
- The random seed is always a function argument, never a module-level constant.