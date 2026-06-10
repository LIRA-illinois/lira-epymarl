#!/bin/bash
echo Total CPU usage
hostnames=$(uname -n | cut -d. -f1)

# Get the CPU load averages
load_avg_1min=$(uptime | awk -F'load average: ' '{print $2}' | cut -d ',' -f1 | tr -d ' ')
load_avg_5min=$(uptime | awk -F'load average: ' '{print $2}' | cut -d ',' -f2 | tr -d ' ')
load_avg_15min=$(uptime| awk -F'load average: ' '{print $2}' | cut -d ',' -f3 | tr -d ' ')

# Get the number of CPU cores
cpu_core=$(nproc --all)

# Calculate percentages
percentage_1min=$(echo "scale=2; ($load_avg_1min / $cpu_core) * 100" | bc)
percentage_5min=$(echo "scale=2; ($load_avg_5min / $cpu_core) * 100" | bc)
percentage_15min=$(echo "scale=2; ($load_avg_15min / $cpu_core) * 100" | bc)

printf "+------------------+-------------------+------------------+-----------------+\n"
printf "| %-16s | %-17s | %-16s | %-14s |\n" "HostName" "Time Intervals" "Load Average" "System Load (%)"
printf "+------------------+-------------------+------------------+-----------------+\n"

printf "| %-16s | %-17s | %-16s | %-15s | \n" "" "1 minute" "$load_avg_1min" "$percentage_1min%"
printf "| %-16s | %-17s | %-16s | %-15s | \n" "$hostnames" "5 minutes" "$load_avg_5min" "$percentage_5min%"
printf "| %-16s | %-17s | %-16s | %-15s | \n" "" "15 minutes" "$load_avg_15min" "$percentage_15min%"

printf "+------------------+-------------------+------------------+-----------------+\n"
