
#!/bin/sh
ps -eo rss,user | awk '{ if ($2 != "USER") { sum[$2] += $1 } } END { print "Memory Usage per User (GB):"; for (i in sum) { print i ": " int(sum[i]/1024/1024) "GB" } }'
