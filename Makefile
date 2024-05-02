.PHONY: quality style test docs utils

check_dirs := tests src examples benchmarks utils

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_dummies.py
	python utils/check_repo.py
	doc-builder style src/accelerate docs/source --max_len 119