# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions
module load gcc/4.8.1
module load slurm 
module load openmpi/gcc-4.8.1/cuda75/1.10.1
module load cuda80/toolkit/8.0
module load python/3.5.1

#-----------------------------------
# Include '.' in PATH
#-----------------------------------
export PATH=.:${PATH}

#####Please Do not Remove the following from your .bashrc Thanks Jason #######
if [ "$(hostname)" == 'kestrel' ]
then    
        module load cmsh
        export EMAIL=`cmsh -c "user show user $USER" | grep email | awk '{print $2 }'`
        export GROUPNAME=`groups $USER | awk '{print $3}'`
        module remove cmsh
fi


##############################################################################

