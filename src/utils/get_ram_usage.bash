#!/bin/bash
# Show system RAM totals via `free` and per-user RAM usage by summing VmRSS from /proc
# Outputs sizes in MiB sorted descending.
set -euo pipefail


# Print system totals
echo Total memory usage
free -h

echo Computing memory usage per user
printf "%-20s %12s %12s\n" "USER" "UID" "RAM (GiB)"
# header printed
# Sum Rss per UID (in kB).
# Prefer /proc/<pid>/smaps_rollup Rss for accurate per-process resident set size;
# fall back to Status:VmRSS when smaps_rollup isn't available.
declare -A uid_rss

for pid_dir in /proc/[0-9]*; do
  pid=${pid_dir#/proc/}
  status_file="$pid_dir/status"
  smaps_rollup="$pid_dir/smaps_rollup"
  if [[ -r "$status_file" ]]; then
    uid_line=$(awk '/^Uid:/ {print $2; exit}' "$status_file" || true)
    rss_kb=0
    if [[ -r "$smaps_rollup" ]]; then
      # Prefer Pss when available; PSS apportions shared pages proportionally.
      pss_kb=$(awk '/^Pss:/ {print $2; exit}' "$smaps_rollup" 2>/dev/null || echo 0)
      if [[ "$pss_kb" -gt 0 ]]; then
        rss_kb=$pss_kb
      else
        # fall back to Rss in smaps_rollup
        rss_kb=$(awk '/^Rss:/ {print $2; exit}' "$smaps_rollup" 2>/dev/null || echo 0)
      fi
    elif [[ -r "$status_file" ]]; then
      # last resort: VmRSS from status (less accurate)
      rss_kb=$(awk '/^VmRSS:/ {print $2; exit}' "$status_file" 2>/dev/null || echo 0)
    else
      rss_kb=0
    fi

    if [[ -n "$uid_line" ]]; then
      uid_rss[$uid_line]=$(( ${uid_rss[$uid_line]:-0} + ${rss_kb:-0} ))
    fi
  fi

done

# Print mapped usernames and sort
for uid in "${!uid_rss[@]}"; do
  user=$(getent passwd "$uid" 2>/dev/null | cut -d: -f1 || echo "UID_$uid")
  rss_gib=$(awk -v kb=${uid_rss[$uid]} 'BEGIN{printf "%.3f", kb/1024/1024}')
  printf "% -20s %12s %12s\n" "$user" "$uid" "$rss_gib"
done | sort -k3 -nr
