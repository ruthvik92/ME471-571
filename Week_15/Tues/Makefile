CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

CFLAGS=-O2

.PHONY: warpdiv dot dotproduct dotproduct2 cg cg_cpu

warpdiv: warpdiv.cu ${objs}
	${NVC} -o warpdiv  ${INCLUDE_DIR} warpdiv.cu ${objs} -lm

dot: dot.cu ${objs}
	${NVC} -o dot  ${INCLUDE_DIR} dot.cu ${objs} -lm

dotproduct: dotproduct.cu ${objs}
	${NVC} -o dotproduct  ${INCLUDE_DIR} dotproduct.cu ${objs} -lm

dotproduct2: dotproduct2.cu ${objs}
	${NVC} -o dotproduct2  ${INCLUDE_DIR} dotproduct2.cu ${objs} -lm

cg: cg.cu ${objs}
	${NVC} -o cg  ${INCLUDE_DIR} ${CFLAGS} -DUSE_GPU cg.cu ${objs} cg_kernels.cu -lm

cg_cpu: cg.cu ${objs}
	${NVC} -o cg_cpu  ${INCLUDE_DIR} ${CFLAGS} cg.cu ${objs} cg_kernels.cu -lm
