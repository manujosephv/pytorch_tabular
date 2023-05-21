# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/manujosephv/pytorch_tabular/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in
  troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to implement
it.

### Write Documentation

Pytorch Tabular could always use more documentation, whether as part of the
official Pytorch Tabular docs, in docstrings, or even on the web in blog
posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/manujosephv/pytorch_tabular/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to
  implement.
- Remember that this is a volunteer-driven project, and that
  contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up PyTorch Tabular for local
development.

```bash
git clone git@github.com:your_name_here/pytorch_tabular.git
```

* Fork the pytorch_tabular repo on GitHub.

* Clone your fork locally and change directory to the checked out folder:

      ```bash
      git clone git@github.com:your_name_here/pytorch_tabular.git
      cd pytorch_tabular
      ```

* Setup a local environment (preferably in a virtual environment). 

      Using python native venv:

      ``` bash
      mkdir .env
      python3 -m venv .env/tabular_env
      source .env/tabular_env/bin/activate
      pip install -e .[dev]
      ```
 
* Create a branch for local development:

      ```bash
      git checkout -b name-of-your-bugfix-or-feature
      ```

      Now you can make your changes locally.

!!! warning

      Never work in the `master` branch!
      
!!! tip
         
      Have meaningful commit messages. This will help with the review and further processing of the PR.

* When you are done, run the `pytest` unit tests and see if everything is a success.
   
      ```bash
      pytest tests/
      ```
!!!note

      If you are adding a new feature, please add a test for it.

* When you are done making changes and all test cases are passing, run `pre-commit` to make sure all the linting and formatting is done correctly.

      ```bash
      pre-commit run --all-files
      ```
   Accept the changes if any after reviewing. 
   
!!!warning   

      Do not commit pre-commit changes to to `setup.cfg`. The file has been excluded from one hook for bump2version compatibility. For a complet and uptodate list of excluded files, please check `.pre-commit-config.yaml` file.

* Commit your changes and push your branch to GitHub:
      ```bash
      git add .
      git commit -m "Your detailed description of your changes."
      git push origin name-of-your-bugfix-or-feature
      ```
* Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
1. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring.

## Tips

To run a subset of tests:

```bash
pytest tests\test_*
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your
changes are committed (including an entry in HISTORY.rst). Then run:

```bash
bump2version patch \# possible: major / minor / patch \$ git push \$
git push --tags
```

GitHub Actions will take care of the rest.
