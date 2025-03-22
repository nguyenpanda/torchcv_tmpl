.PHONY: build
build:
	python -m build

.PHONY: wheel
wheel: build
	python -m twine upload --repository testpypi dist/*

.PHONY: test_pypi
test_pypi: wheel

.PHONY: wheel
pypi: wheel
