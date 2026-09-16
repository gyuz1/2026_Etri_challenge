#!/usr/bin/env bash
# Explicit package sync, including all inherited configs. Never launches training.
set -euo pipefail
source "$(dirname "$0")/_goal_anchor_pair.sh"
cd "$PAIR_ROOT"
case "${1:---dry-run}" in
  --dry-run)
    echo "Would synchronize Python/shell sources under projects/mmdet3d_plugin, projects/configs, tools, scripts."
    echo "Destination: $PAIR_REMOTE_HOST:$PAIR_REMOTE_ROOT (A5000)."
    echo 'No data, checkpoints, .git, deletion of extra remote files, or remote access.'
    echo 'Use --sync only after existing jobs finish; busy GPUs/training refuse synchronization.'
    exit 0 ;;
  --sync) [ "$#" = 1 ] || exit 2 ;;
  *) echo 'usage: bash scripts/sync_goal_anchor_pair.sh [--dry-run|--sync]' >&2; exit 2 ;;
esac
pair_require_idle a5000
stage=$(mktemp -d /tmp/law_goal_anchor_sync.XXXXXX)
pair_source_files > "$stage/files.txt"
pair_manifest > "$stage/source_manifest.sha256"
tar -czf "$stage/source.tar.gz" -T "$stage/files.txt"
remote_stage=$(pair_ssh 'mktemp -d /tmp/law_goal_anchor_sync.XXXXXX')
case "$remote_stage" in /tmp/law_goal_anchor_sync.*) ;; *) echo 'Unexpected remote staging path' >&2; exit 1 ;; esac
scp -P "$PAIR_REMOTE_PORT" "$stage/source.tar.gz" "$stage/files.txt" "$stage/source_manifest.sha256" \
  "$PAIR_REMOTE_HOST:$remote_stage/"
pair_require_idle a5000
# Keep a recoverable backup of existing targets. No source deletion is performed.
printf -v remote_command '%q ' bash -s -- "$remote_stage" "$PAIR_REMOTE_ROOT"
pair_ssh "$remote_command" <<'REMOTE'
set -euo pipefail
stage=$1; root=$2
test -d "$root"
mkdir "$stage/unpacked"
tar -xzf "$stage/source.tar.gz" -C "$stage/unpacked"
(cd "$stage/unpacked" && sha256sum --check --strict "$stage/source_manifest.sha256")
while IFS= read -r path; do
  if [ -f "$root/$path" ]; then printf '%s\n' "$path"; fi
done < "$stage/files.txt" > "$stage/existing.txt"
tar -czf "$stage/before_sync.tar.gz" -C "$root" -T "$stage/existing.txt"
tar -xzf "$stage/source.tar.gz" -C "$root"
(cd "$root" && sha256sum --check --strict "$stage/source_manifest.sha256")
printf 'Source sync verified. Recoverable pre-sync backup: %s/before_sync.tar.gz\n' "$stage"
REMOTE
pair_verify_source a5000
echo "Local package retained at $stage. No training started."
