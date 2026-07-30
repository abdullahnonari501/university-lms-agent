#!/usr/bin/env bash
# Print the current public URL.
#
# Quick tunnels mint a NEW hostname every time cloudflared restarts -- including
# on reboot -- so the URL cannot be written down anywhere permanent. A named
# tunnel (needs a Cloudflare account) is the only way to get a stable address.
url=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/cloudflared.log 2>/dev/null | tail -1)
[ -z "$url" ] && url=$(journalctl --user -u cloudflared -n 500 --no-pager 2>/dev/null \
                      | grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" | tail -1)
if [ -z "$url" ]; then
  echo "no tunnel URL found -- is cloudflared.service running?" >&2
  systemctl --user is-active cloudflared.service >&2
  exit 1
fi
echo "$url"
