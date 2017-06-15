#########################################################################
# File: make.sh
# Author: Naiyong Ao
# Email: aonaiyong@gmail.com
# Time: Tue 03 Feb 2015 06:59:07 PM CST
#########################################################################
#!/bin/bash

codedir="/media/indexDisk/naiyong/VLDB_1Range+2Ranges_Backup/VLDB_2Ranges_Backup/VLDB_CODE_2Ranges_Final/Compressed/src/"
cd $codedir
for outerdir in `ls`
do
	cd $outerdir
	for innerdir in `ls`
	do
		cd $innerdir
		make clean
#		make
		echo ''
		cd ..
	done
	cd ..
	echo ''
done
