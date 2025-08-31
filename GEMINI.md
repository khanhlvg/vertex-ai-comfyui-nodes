# Build

- After making code changes, run `ruff check . --fix --exclude=tmp` to ensure there are no linting errors.

# Code management

- Ignore the folder `tmp/`. I use it to store temporary files for Gemini CLI to read but I don't want it to be added to the git repo. I also don't want to add them into .gitignore