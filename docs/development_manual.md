# Bump Version and Trigger Build

## Bump Version(Patch Update)

```bash
#Commit all changes
bump2version patch
```

- eg: 0.1.1 to 0.1.2-dev0
- For minor patch updates
- No tags are created

## Bump Version(Minor Update)

```bash
#Commit all changes
bump2version minor
```

- eg: 0.1.1-dev0 to 0.2.1-dev0
- For minor feature updates
- No tags are created

## Bump Version(Major Update)

```bash
#Commit all changes
bump2version major
```

- eg: 0.1.1-dev0 to 1.0.0-dev0
- For major feature updates
- No tags are created

## Bump Version(Release)

```bash
# Add new Version and changelog to History.md
# Commit all changes and run
bump2version --tag release
# Check if the tag is present
git tag
# Push the changes to GitHub
git push origin <tag_name>

```

- eg: 0.1.1-dev0 to 0.1.1
- To trigger GitHub Actions to push to PyPi
- Tags are created

# Revert Version and Delete a Tag

- Update the version numbers in
  1. setup.py
  1. setup.cfg
  1. __init__.py
- Delete the Git Tags in local
  ```bash
  git tag -d <tag_name>
  ```
- Delete the tags from GitHub
  ```bash
  git push --delete origin <tag_name>
  ```
