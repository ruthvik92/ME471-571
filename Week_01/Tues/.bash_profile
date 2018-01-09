# Some useful bash shell configurations. 
# This file is run every time you run a start a new Terminal window

echo "Hello! Running .bash_profile"

# Set up colors for the directory command
alias ls='ls -FG'

PS1='\[\e[0;34;1m\] \w % \[\e[0;1m\]'
export PS1

ANACONDA=/usr/local/Anaconda3
export ANACONDA

PATH=${ANACONDA}/bin:${PATH}
export PATH