# Authenticate Travis

## Encrypt Password
```
travis encrypt <password>
```
- Copy the resuting encrypted password and add to travis.yml

# Bump Version and Trigger Build

## Bump Version(Patch Update)

```
#Commit all changes
bump2version patch
```

- eg: 0.1.1 to 0.1.2-dev0
- For minor patch updates
- No tags are created

## Bump Version(Minor Update)

```
#Commit all changes
bump2version minor
```

- eg: 0.1.1-dev0 to 0.2.1-dev0
- For minor feature updates
- No tags are created

## Bump Version(Major Update)

```
#Commit all changes
bump2version major
```

- eg: 0.1.1-dev0 to 1.0.0-dev0
- For major feature updates
- No tags are created

## Bump Version(Release)

```
# Add new Version and changelog to History.md
# Commit all changes
bump2version --tag release

git push
git push --tags

```

- eg: 0.1.1-dev0 to 0.1.1
- To trigger Travis CI build
- Tags are created

# Revert Version and Delete a Tag

- Update the version numbers in 
    1. setup.py 
    2. setup.cfg 
    3. __init__.py
- Delete the Git Tags in local
    ```
    git tag -d <tag_name>
    ```
- Delete the tags from GitHub
    ```
    git push --delte origin <tag_name>
    ```
