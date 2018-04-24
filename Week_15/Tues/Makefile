CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: warpdiv dot dotproduct

warpdiv: warpdiv.cu ${objs}
	${NVC} -o warpdiv  ${INCLUDE_DIR} warpdiv.cu ${objs} -lm

dot: dot.cu ${objs}
	${NVC} -o dot  ${INCLUDE_DIR} dot.cu ${objs} -lm

dotproduct: dotproduct.cu ${objs}
	${NVC} -o dotproduct  ${INCLUDE_DIR} dotproduct.cu ${objs} -lm
