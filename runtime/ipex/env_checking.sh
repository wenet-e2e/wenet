#!/usr/bin/bash

has_issue=0
echo "**********CPU Information************"
cpu_type=$(lscpu | grep "Vendor ID"|awk -F':' '{print $2}'|xargs)
if [[ ${cpu_type} != "GenuineIntel" ]]; then
echo "Unknown Device, environment checking does not ensure correct."
exit
fi
pyhsical_number=0
core_number=0
logical_number=0
ht_number=0
logical_number=$(grep "processor" /proc/cpuinfo|sort -u|wc -l)
physical_number=$(grep "physical id" /proc/cpuinfo|sort -u|wc -l)
core_number=$(grep "cpu cores" /proc/cpuinfo|uniq|awk -F':' '{print $2}'|xargs)
ht_number=$((logical_number / (physical_number * core_number)))
echo "Physical CPU Number : ${physical_number}"
echo "CPU Core Number : ${core_number}"
echo "Logical CPU Number : ${logical_number}"
echo "Hyper Thread Number : ${ht_number}"
if [[ `expr ${ht_number} \< 2 | bc` -eq 1 ]]; then
has_issue=1
echo "Warning: Please turn on Hyper-Threading option in BIOS"
else
echo "CPU Configuration: OK"
fi
echo "************Memory Information*******"
mem_speed=($(dmidecode -t memory | grep 'Configured Memory Speed:' | grep 'MT' | awk -F':' '{print $2}'| awk -F' ' '{print $1}'|xargs))
mem_num=${#mem_speed[@]}
model=$(lscpu| grep "Model:" | awk -F':' '{print $2}' | xargs)
mem_need=0
if [[ $model == 106 ]] || [[ $model == 143 ]];then
mem_need=$((${physical_number} * 8))
elif [[ $model == 85 ]]; then
mem_need=$((${physical_number} * 6))
fi
if [[ ${mem_num} != ${mem_need} ]]; then
has_issue=1
echo "Warning: Lack of memory"
else
echo "Memory Number: ${mem_num}"
echo "Memory Speed: ${mem_speed[0]} MT/s"
echo "Memory Number: OK"
fi
mem_type=($(dmidecode -t memory | grep 'Configured Memory Speed:' | grep 'MT' | awk -F':' '{print $2}'| awk -F' ' '{print $1}'|uniq|xargs))
mem_type_num=${#mem_type[@]}
if [[ `expr ${mem_type_num} \> 1}|bc` -eq 1 ]]; then
has_issue=0
echo "Memory Type: ${mem_type}"
echo "Warning: Memory Type is not consistent: ${mem_type}"
else
echo "Memory Consistency: OK"
fi
echo "************Swap Information*********"
swap_used=$(swapon -s | grep swapfile | awk -F' ' '{print $4}'|xargs)
if [[ `expr ${swap_used} \> 0 | bc` -eq 1 ]]; then
has_issue=1
echo "Warning: please clean swap memory"
else
echo "Swap: OK"
fi

echo "********CPU Usage Information********"
cpu_used=$(top -bn 1 -i -c |grep Cpu | awk -F',' '{print $1}' | grep -oP '\d*\.\d+')
if [[ `expr ${cpu_used} \> 5.0 | bc` -eq 1 ]]; then
has_issue=1
echo "Warning: There are other processes running!!"
else
echo "CPU Status: OK"
fi

if [[ $model == 143 ]]; then
echo "*****************Kernel Information********"
kernel_version=$(uname -r)
echo "Current Kernel Version: ${kernel_version}"
kernel_versions=($(uname -r | awk '{len=split($0,a,".");for(i=1;i<=len;i++) print a[i]}'|xargs))
kernel_main=${kernel_versions[0]}
kernel_mid=${kernel_versions[1]}
if [[ `expr ${kernel_main} \< 5 | bc` -eq 1 ]]; then
has_issue=1
echo "Error: kernel version is lower than 5.16, please upgrade kernel version"
elif [[ `expr ${kernel_mid} \< 16 | bc` -eq 1 ]]; then
echo "Error: Kernel version is lower then 5.16, please upgrade kernel version"
else
echo "Kernel Version: OK"
fi
fi

if [[ ${has_issue} == 1 ]]; then
echo "*********Friendly Reminder***********"
echo "Please process error & warning before you start benchmark!"
else
echo "*********Hardware Verify Passed******"
echo "Your hardware environment is ready, good luck with your benchmarking."
fi

