#!/usr/bin/env bash
echo "### GPU brief (util/clock/power/pstate)"
nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv -i 0

echo
echo "### GPU processes"
nvidia-smi

echo
echo "### GPU dmon (5s)"
nvidia-smi dmon -s pucvmt -d 1 -c 5

echo
echo "### Who uses /dev/nvidia*"
fuser -v /dev/nvidia* 2>/dev/null || true

echo
echo "### CPU governor + avg MHz"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
grep "MHz" /proc/cpuinfo | awk '{sum+=$4} END {print "avg MHz:",sum/NR}'

echo
echo "### PCIe link (需要 root)"
sudo lspci -vv -s 02:00.0 | egrep -i "LnkCap|LnkSta" || true

echo
echo "### Recent NVIDIA driver messages"
dmesg | tail -n 50 | egrep -i "nvrm|nvidia|xid" || true
