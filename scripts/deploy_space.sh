#!/usr/bin/env bash
# Deploys the current `main` branch to the Hugging Face Space as one snapshot commit.
# The Space gets main's files without git history and without sample_images/:
# the Hub rejects binary files sent over plain git, and the app doesn't need them.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

tmp_index=$(mktemp -u)
trap 'rm -f "$tmp_index"' EXIT

GIT_INDEX_FILE=$tmp_index git read-tree main
GIT_INDEX_FILE=$tmp_index git rm --cached -r -q --ignore-unmatch sample_images
tree=$(GIT_INDEX_FILE=$tmp_index git write-tree)
commit=$(git commit-tree "$tree" -m "Deploy $(git rev-parse --short main)")

git -c credential.helper= \
    -c 'credential.helper=!f() { echo username=mgeorgiev5; echo "password=$(cat ~/.cache/huggingface/token)"; }; f' \
    push space "$commit:refs/heads/main" --force
