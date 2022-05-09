#!/bin/bash

# Download datasets
for dset in libri vctk; do
  for suff in dev test; do
    printf "${GREEN}\nStage 0: Downloading ${dset}_${suff} set...${NC}\n"
    adapted_from_vpc/download_data.sh ${dset}_${suff} || exit 1;
   done
done
