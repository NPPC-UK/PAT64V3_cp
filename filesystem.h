#ifndef FILESYSTEM
#define FILESYSTEM
#include <regex>
#include <vector>

#include <boost/filesystem.hpp>

/** 
 * \brief Lists the contents of a directory matching a pattern
 * */
std::vector<boost::filesystem::path> getFiles(
    boost::filesystem::path,
    std::regex); 
#endif
