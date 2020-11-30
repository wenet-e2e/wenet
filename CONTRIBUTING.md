# Contributing guidelines

## Pre-commit tidy/linting hook

You'll need to install flake8 first.

We use flake8 to perform additional formatting and semantic checking of code.
We provide a pre-commit git hook for performing these checks, before a commit
is created:

```bash
ln -s ../../tools/git-pre-commit .git/hooks/pre-commit
```
