#include <regex>
#include <iostream>

#include <boost/filesystem.hpp>

#include "filesystem.h"

int main(int argc, char* argv[]) {
  boost::filesystem::path p = boost::filesystem::path(argv[1]);
  std::vector<boost::filesystem::path> v = getFiles(p, std::regex(".*\\.cpp"));


  for (std::vector<boost::filesystem::path>::iterator it = v.begin();
      it != v.end();
      ++it) {

    std::cout << *it << '\n';
  }

  return 0;
}
