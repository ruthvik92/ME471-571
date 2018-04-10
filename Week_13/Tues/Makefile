CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: printCoordinates matrix mapping_demo

printCoordinates: printCoordinates.cu ${objs}
	${NVC} -o printCoordinates  ${INCLUDE_DIR} printCoordinates.cu ${objs} -lm

matrix: matrix.cu ${objs}
	${NVC} -o matrix  ${INCLUDE_DIR} matrix.cu ${objs} -lm

mapping_demo: mapping_demo.cu ${objs}
	${NVC} -o mapping_demo  ${INCLUDE_DIR} mapping_demo.cu ${objs} -lm
