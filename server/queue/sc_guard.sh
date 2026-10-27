#!/usr/bin/env bash
# restart any unfinished Selective Context shard whose process has died
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025; cd $W
while true; do left=0
  for i in $(seq 0 15); do grep -q SHARD_DONE runs/rv5/sc/log$i.txt && continue; left=1
    pgrep -f "sc_pack.py $i 16" > /dev/null || pgrep -f "retry_pack.sh . .* sc_pack.py $i 16" > /dev/null || nohup bash queue/retry_pack.sh $((1 + i % 2)) $W/extra_pkgs_sc sc_pack.py $i 16 runs/rv5/sc/log$i.txt > /dev/null 2>&1 &
  done; [ $left = 0 ] && exit; sleep 300; done
