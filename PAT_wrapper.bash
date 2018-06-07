#!/bin/bash

fnames=$(find /mnt/tmp2/experiments/W13 -name '*_VIS_sv*') 

for f in $fnames; do
  echo "Processing ${f}"
  plant_id=$(echo $f | awk -F "/" '{print $6}')
  date=$(echo $f | awk -F "/" '{print $7}')
  name=$(echo $f | awk -F "/" '{print $8}')
  name=${name#00_VIS_sv_}
  angle=${name%-0-0-0.png}
  outname=${plant_id}_${date}_${angle}.jpg

  ./PAT32_wb --input-file $f \
    --output-file /mnt/tmp/W13/output/$outname \
    --date $date \
    --plant-id $plant_id >> /mnt/tmp/W13/output/output.csv
done
