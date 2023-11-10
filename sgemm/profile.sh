#!/bin/bash

ncu -o sgemm_naive --set full --kernel-id ::sgemm_naive:6 sgemm
ncu -o sgemm_global_mem_coalesce --set full --kernel-id ::sgemm_global_mem_coalesce:6 sgemm
ncu -o sgemm_shared_mem_block --set full --kernel-id ::sgemm_shared_mem_block:6 sgemm
ncu -o sgemmBlocktiling --set full --kernel-id ::sgemmBlocktiling:6 sgemm
ncu -o sgemmVectorize --set full --kernel-id ::sgemmVectorize:6 sgemm
ncu -o sgemmVectorize_double_buffering --set full --kernel-id ::sgemmVectorize_double_buffering:6 sgemm
ncu -o sgemmVectorize_double_buffering2 --set full --kernel-id ::sgemmVectorize_double_buffering2:6 sgemm
ncu -o sgemmWarptiling --set full --kernel-id ::sgemmWarptiling:6 sgemm
ncu -o sgemmWarptiling2 --set full --kernel-id ::sgemmWarptiling2:6 sgemm
ncu -o Sgemm_kun --set full --kernel-id ::Sgemm_kun:6 sgemm
ncu -o Sgemm_kun_v3 --set full --kernel-id ::Sgemm_kun_v3:6 sgemm
