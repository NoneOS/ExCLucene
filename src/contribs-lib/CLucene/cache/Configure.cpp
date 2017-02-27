/*
 * Configure.cpp
 *
 */

#include "Configure.h"

CL_NS_USE2(search, cache)

Configure::Configure() {
	// TODO Auto-generated constructor stub
}

Configure::~Configure() {
	// TODO Auto-generated destructor stub
}

std::string Configure::QRC_RESULT_CACHE = "QRC_RESULT_CACHE";
std::string Configure::PLC_POSTINGLIST_CACHE = "PLC_POSTINGLIST_CACHE";
std::string Configure::DC_DOCUMENT_CACHE = "DC_DOCUMENT_CACHE";
std::string Configure::SC_SNIPPET_CACHE = "SC_SNIPPET_CACHE";

const TCHAR* Configure::ITEM_ID = L"id";	// doc id, used for referring DB, primary key
const TCHAR* Configure::ITEM_URL = L"url";
const TCHAR* Configure::ITEM_TITLE = L"title";
const TCHAR* Configure::ITEM_CONTENT = L"content";
const TCHAR* Configure::ITEM_PATH = L"path";	// file path of the original item
