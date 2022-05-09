#!/bin/sh
# ------- Use sox to downsample
# Before use, please specify SOXPATH
# Usage: sub_down_sample.sh input_wav output_wav target_sampling_rate
#
#
source $PWD/env.sh

SOXPATH=sox
echo ${SOXPATH} $1 -b 16 $2 rate -I $3
${SOXPATH} $1 -b 16 $2 rate -I $3

