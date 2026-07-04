#!/usr/bin/env bash
# Reconcile the repository's labels with .github/labels.yml.
#
# Idempotent: creates labels that are missing and updates the color/description
# of labels that have drifted. Never deletes labels (a label may be in use).
#
# Usage:
#   GH_TOKEN=<token> bash .github/setup_labels.sh            # current repo
#   GH_REPO=owner/name bash .github/setup_labels.sh          # explicit repo
#
# Requires the `gh` CLI (authenticated) and `python3` (stdlib only) to parse the
# YAML — no third-party dependencies.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
labels_file="${here}/labels.yml"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI is required" >&2
  exit 1
fi

# Emit "name<TAB>color<TAB>description" for each entry in labels.yml.
parse_labels() {
  python3 - "$labels_file" <<'PY'
import sys, re

name = color = desc = None
out = []

def flush():
    if name is not None:
        out.append("\t".join([name, color or "cccccc", desc or ""]))

for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.rstrip("\n")
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    m = re.match(r"-\s+name:\s*(.+)$", stripped)
    if m:
        flush()
        name = m.group(1).strip().strip('"').strip("'")
        color, desc = None, None
        continue
    m = re.match(r"color:\s*(.+)$", stripped)
    if m:
        color = m.group(1).strip().strip('"').strip("'").lstrip("#")
        continue
    m = re.match(r"description:\s*(.+)$", stripped)
    if m:
        desc = m.group(1).strip().strip('"').strip("'")
        continue

flush()
print("\n".join(out))
PY
}

count=0
while IFS=$'\t' read -r name color desc; do
  [ -z "$name" ] && continue
  # `gh label create --force` upserts: creates if absent, updates otherwise.
  gh label create "$name" --color "$color" --description "$desc" --force
  count=$((count + 1))
done < <(parse_labels)

echo "Synced ${count} labels from ${labels_file}."
