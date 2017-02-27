/*
 * example.cpp
 *
 */
#include "BMWQuery.h"
#include "ListPtr_B.h"
#include "common.h"
#include "type.h"
#include "WandQuery.h"
#include "ListPtr.h"
#include "factory.h"

int main(){

	//give a index address
	const string index_dir = "/wand";			// wand index path, including two files
	const string maxscore_path = index_dir +  ".maxscore";	//  file 1: maxscore of terms
	const string query_path = index_dir +  ".query";	//  file 2: text file, each line is a sequence of terms
	const string res_path = "./resDir/";			//  empty directory, used for store query result

	QueryBase &query = * Factory::getFromName("wand");	// for wand query
	//QueryBase &query = * Factory::getFromName("bmw");	// for bmw query
	query.loadMaxScore(maxscore_path);
	query.loadQuerySet(query_path);
	query.querySet_process(res_path);

	return 0;
}


