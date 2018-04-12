# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions
module load gcc/4.8.1
module load slurm 
module load openmpi/gcc-4.8.1/cuda75/1.10.1
module load cuda75/toolkit/7.5
module load python/3.5.1

source ~/.bash_alias
export PATH=.:${PATH}

