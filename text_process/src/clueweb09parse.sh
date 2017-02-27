#########################################################################
# File Name: parse.sh
#########################################################################
#!/bin/bash

input="/home/yebenjun/dataset/clueweb09/"

if [ $# -lt 2 ];then
    echo "please provide at least two arguments."
else
for((i=$1;i<=$2;i++))
do
    subfolderName=`printf "%sen%04d" $input $i`
    echo "gunzip -r $subfolderName"
    `gunzip -r $subfolderName`
    java -cp .:../jsoup-1.8.2.jar -ea CluewebParse $i
    echo "rm -rf $subfolderName"
    `rm -rf $subfolderName`
done
fi
