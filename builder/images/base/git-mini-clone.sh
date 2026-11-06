#!/bin/bash
set -xe
REPO="$1"
REF="$2"
DEST="$3"
git init "$DEST"
git -C "$DEST" remote add origin "$REPO"

retry-tool git -C "$DEST" fetch --depth=1 origin "$REF"

git -C "$DEST" config advice.detachedHead false
git -C "$DEST" checkout FETCH_HEAD

# Streamarr: record the revision this clone resolved to, so that every archive names the exact
# source of each clone, including one whose recipe passes no revision. One file per clone keeps
# the records of all build stages when their prefixes are merged into the final image.
record_dir="${FFBUILD_PREFIX:-/opt/ffbuild}/share/streamarr-build-sources"
mkdir -p "$record_dir"
printf '%s %s %s\n' "$(basename "$DEST")" "$REPO" "$(git -C "$DEST" rev-parse HEAD)" \
    > "$record_dir/$(basename "$DEST")-$(printf '%s' "$REPO" | cksum | cut -d' ' -f1).txt"
