# Contributing to ColorCast

Contributions are welcome. Here is the preferred workflow.

## Development Setup

```bash
git clone https://github.com/MichailSemoglou/ColorCast.git
cd ColorCast
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"
```

## Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Make your changes. Follow the existing code style (black, isort,
   ruff). Add tests for new functionality.
4. Run the full quality suite locally before pushing:

   ```bash
   pytest
   black --check colorcast tests
   isort --check-only colorcast tests
   ruff check colorcast tests
   mypy colorcast
   ```

5. Push your branch and open a pull request against `main`.
6. A CI run will execute automatically on your PR. All checks must pass
   before merge.

## Code Conventions

- **Formatter:** black (line length 100)
- **Import order:** isort (black profile)
- **Linter:** ruff (rules E, F, I, W, UP, B, SIM, C4)
- **Type checker:** mypy (Python 3.10+)
- **American English** spelling in prose; straight ASCII in docstrings
  and code comments.

## Testing

- Run `pytest` to execute the full test suite.
- New features should include tests.
- Property-based tests (Hypothesis) are welcome for algorithmic
  invariants.
- GUI tests require `QT_QPA_PLATFORM=offscreen`.

## Reporting Issues

Use the GitHub issue tracker. Choose the bug report or feature request
template when available.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
