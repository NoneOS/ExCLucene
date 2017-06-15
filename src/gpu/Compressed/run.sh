#########################################################################
# File: run.sh
# Author: Naiyong Ao
# Email: aonaiyong@gmail.com
# Time: Tue 03 Feb 2015 07:18:15 PM CST
#########################################################################
#!/bin/bash

index_dir="/media/indexDisk/naiyong/dataset/"
code_dir="/media/indexDisk/naiyong/VLDB_1Range+2Ranges_Backup/VLDB_2Ranges_Backup/VLDB_CODE_2Ranges_Final/Compressed/src/"
data_dir="/media/indexDisk/naiyong/data/"


./make.sh
for dataset in `ls $index_dir`
do
	echo $dataset

#	# NewPFD
#	cd $code_dir
#	cd "NewPFD"

	# Compression
#	cd "NewPFD_Compression"
#	./NewPFD_Compression $dataset
#	cd ..

	# Decompression
#	cd "NewPFD_Decompression"
#	./NewPFD_Decompression $dataset
#	cd ..

	# erasing
#	cd $data_dir
#	cd "NewPFD/Compression"
#	rm -rf */*

#	echo ''


	# ParaPFD
#	cd $code_dir
#	cd "ParaPFD"

	# Compression
#	cd "ParaPFD_Compression"
#	./ParaPFD_Compression $dataset
#	cd ..

	# Decompression
#	cd "ParaPFD_Decompression"
#	./ParaPFD_Decompression $dataset
#	cd ..

	# Intersection
#	cd "ParaPFD_Intersection_NoFRAC"
#	./ParaPFD_Intersection $dataset
#	cd ..

#	cd "ParaPFD_Intersection_FRAC"
#	./ParaPFD_Intersection $dataset
#	cd ..

	# erasing
#	cd $data_dir
#	cd "ParaPFD/Compression"
#	rm -rf */*

#	echo ''


	# LRC
#	cd $code_dir
#	cd "LRC"

	# Generator
#	cd "LRC_Generator"
#	./LRC_Generator $dataset
#	cd ..

	# Compression
#	cd "LRC_Compression"
#	./LRC_Compression $dataset
#	cd ..

	# Decompression
#	cd "LRC_Decompression"
#	./LRC_Decompression $dataset
#	cd ..

	# Intersection
#	cd "LRC_Intersection"
#	./LRC_Intersection $dataset
#	cd ..

	# erasing
#	cd $data_dir
#	cd "LRC/Compression"
#	rm -rf */*

#	echo ''


	# LRCSeg
#	cd $code_dir
#	cd "LRCSeg"

	# Generator
#	cd "LRCSeg_Generator"
#	./LRCSeg_Generator $dataset
#	cd ..

	# Compression
#	cd "LRCSeg_Compression"
#	./LRCSeg_Compression $dataset
#	cd ..

	# Decompression
#	cd "LRCSeg_Decompression"
#	./LRCSeg_Decompression $dataset
#	cd ..

	# Intersection
#	cd "LRCSeg_Intersection"
#	./LRCSeg_Intersection $dataset
#	cd ..

	# erasing
#	cd $data_dir
#	cd "LRCSeg/Compression"
#	rm -rf */*

#	echo ''


	# SegLRC
	cd $code_dir
	cd "SegLRC"

	# Generator
	cd "SegLRC_Generator"
	./SegLRC_Generator $dataset
	cd ..

	# Compression
	cd "SegLRC_Compression"
	./SegLRC_Compression $dataset
	cd ..

	# Decompression
#	cd "SegLRC_Decompression"
#	./SegLRC_Decompression $dataset
#	cd ..

	# Intersection
	cd "SegLRC_Intersection"
	./SegLRC_Intersection $dataset
	cd ..

	# erasing
	cd $data_dir
	cd "SegLRC/Compression"
	rm -rf */*

	echo ''


	# HS256_SegLRC
	cd $code_dir
	cd "HS_SegLRC"

	# Generator
	cd "HS_SegLRC_Generator"
	./HS_SegLRC_Generator $dataset 256
	cd ..

	# Compression
	cd "HS_SegLRC_Compression"
	./HS_SegLRC_Compression $dataset 256
	cd ..

	# Decompression
#	cd "HS_SegLRC_Decompression"
#	./HS_SegLRC_Decompression $dataset 256
#	cd ..

	# Intersection
	cd "HS_SegLRC_Intersection"
	./HS_SegLRC_Intersection $dataset 256
	cd ..

	# erasing
	cd $data_dir
	cd "HS_SegLRC/Compression"
	rm -rf */*

	echo ''


	# HS128_SegLRC
#	cd $code_dir
#	cd "HS_SegLRC"

	# Generator
#	cd "HS_SegLRC_Generator"
#	./HS_SegLRC_Generator $dataset 128
#	cd ..

	# Compression
#	cd "HS_SegLRC_Compression"
#	./HS_SegLRC_Compression $dataset 128
#	cd ..

	# Decompression
#	cd "HS_SegLRC_Decompression"
#	./HS_SegLRC_Decompression $dataset 128
#	cd ..

	# Intersection
#	cd "HS_SegLRC_Intersection"
#	./HS_SegLRC_Intersection $dataset 128
#	cd ..

	# erasing
#	cd $data_dir
#	cd "HS_SegLRC/Compression"
#	rm -rf */*

#	echo ''

	
	# HS64_SegLRC
#	cd $code_dir
#	cd "HS_SegLRC"

	# Generator
#	cd "HS_SegLRC_Generator"
#	./HS_SegLRC_Generator $dataset 64
#	cd ..

	# Compression
#	cd "HS_SegLRC_Compression"
#	./HS_SegLRC_Compression $dataset 64
#	cd ..

	# Decompression
#	cd "HS_SegLRC_Decompression"
#	./HS_SegLRC_Decompression $dataset 64
#	cd ..

	# Intersection
#	cd "HS_SegLRC_Intersection"
#	./HS_SegLRC_Intersection $dataset 64
#	cd ..

	# erasing
#	cd $data_dir
#	cd "HS_SegLRC/Compression"
#	rm -rf */*

#	echo ''

done
