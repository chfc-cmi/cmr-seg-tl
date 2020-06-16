#!/usr/bin/env bash

dos2unix $1
# quick and dirty contour extraction one liner
perl -ne 'chomp;if($current && / /){print "$_ $current\n"} else {if($skip){$skip=0;}else{$current=0;}} if($newstart){$current=$_;$newstart=0;$skip=1};if(/XYCONTOUR/){$newstart = 1}' $1 >$2
