#!/usr/bin/env bash
# A copy of a model-data bundle with the commodity partition (scripts/build_commodity_partition.py)
# in place of the co-purchase one.  Every other file is a symbolic link to the source bundle,
# which is left untouched; the copy gets its own ragged index and fingerprint, so checkpoints
# fitted on it cannot be mixed up with the source's.
#   SRC     the source bundle (required)
#   DEST    the new bundle root (required)
#   COLUMN  COMMODITY_DESC (default) or SUB_COMMODITY_DESC
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(cd "${SRC:?set SRC to the source bundle}" && pwd -P)
DEST=${DEST:?set DEST to the new bundle root}
COLUMN=${COLUMN:-COMMODITY_DESC}
[[ -e $DEST ]] && { echo "$DEST exists" >&2; exit 1; }
mkdir -p "$DEST/data" "$DEST/basket_input"
for f in "$SRC"/data/*; do ln -s "$f" "$DEST/data/"; done
for f in "$SRC"/basket_input/*; do
  case $(basename "$f") in
    items_affinity.parquet|affinity_manifest.json|model_data_fingerprint.json|v3_index_affinity.npz) ;;
    *) ln -s "$f" "$DEST/basket_input/" ;;
  esac
done
python3 -u scripts/build_commodity_partition.py --data-root "$DEST" --column "$COLUMN"
python3 -u data_prep/fingerprint.py --data-root "$DEST" --core-only   # builds the ragged index first
