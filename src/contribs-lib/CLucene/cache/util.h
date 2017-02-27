/*
 * util.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_UTIL_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_UTIL_H_

/**
 * This head file utilizes (w)stringstream in CXX to convert int-base/float type
 * into (w)string. Since CL had provided other related methods, maybe we should 
 * keep the same layout, refers to "shared/config/repl_tchar.h" or 
 * "shared/util/Misc.h"
 */

#include "CLucene/StdHeader.h"
#include <sstream>
#include <iomanip>

CL_NS_DEF2(search, cache)

template<typename T>
inline static std::string ToString(const T &ia) {
    std::stringstream ss;
    ss << ia;
    std::string str;
    ss >> str;
    return str;
}

template<typename T>
inline static std::wstring ToWString(const T &ia) {
    std::wstringstream ss;
    ss << ia;
    std::wstring str;
    ss >> str;
    return str;
}

inline static std::string floatToString(const float &d) {
    std::stringstream ss;
    ss << std::setprecision(6) << d;
    std::string str;
    ss >> str;
    return str;
}

inline static std::string doubleToString(const double &d) {
    std::stringstream ss;
    ss << std::setprecision(6) << d;
    std::string str;
    ss >> str;
    return str;
}

// maybe using STRCPY_AtoW & STRCPY_WtoA instead
inline static std::wstring StringToWString(const std::string &str) {
    std::wstring wstr(str.length(), L' ');
    std::copy(str.begin(), str.end(), wstr.begin());
    return wstr;
}

inline static std::string WStringToString(const std::wstring &wstr) {
    std::string str(wstr.length(), ' ');
    std::copy(wstr.begin(), wstr.end(), str.begin());
    return str;
}

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_UTIL_H_ */
