CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: mapping_demo2 pool

mapping_demo2: mapping_demo2.cu ${objs}
	${NVC} -o mapping_demo2  ${INCLUDE_DIR} mapping_demo2.cu ${objs} -lm

pool: pool.cu ${objs}
	${NVC} -o pool  ${INCLUDE_DIR} pool.cu ${objs} -lm
