"""Project source package.

Per ADR-009 (hybrid notebook-first), most pipeline code lives in
``notebooks/00_main.ipynb``. Only modules that need CI-level unit tests
(currently ``preprocessing``) are kept as standalone Python files here.
"""
