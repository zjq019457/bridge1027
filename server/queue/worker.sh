#!/usr/bin/env bash
# usage: worker.sh GPU ; pops lines "name<TAB>command" from queue/jobs.txt
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025; G=$1
while true; do
  line=$(flock $W/queue/lock bash -c "head -1 $W/queue/jobs.txt; sed -i 1d $W/queue/jobs.txt")
  [ -z "$line" ] && { echo "[$(date -Is)] WORKER_DONE gpu=$G"; exit; }
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  for i in $(seq 1 600); do
    free=$(nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits | tr -d " ")
    [ "$free" -ge 20000 ] && break; sleep 30
  done
  echo "[$(date -Is)] start $name gpu=$G"
  CUDA_VISIBLE_DEVICES=$G bash -c "$cmd" > $W/queue/$name.log 2>&1 < /dev/null
  rc=$?; echo "[$(date -Is)] end $name rc=$rc"
  if [ $rc -ne 0 ] && grep -q "out of memory" $W/queue/$name.log && [ ! -f $W/queue/$name.retried ]; then
    touch $W/queue/$name.retried; flock $W/queue/lock bash -c "printf \"%s\n\" \"$line\" >> $W/queue/jobs.txt"; echo "requeued $name"
  fi
done
