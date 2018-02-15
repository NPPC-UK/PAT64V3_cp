#include <iterator>

#include "filesystem.h"

using std::vector;
using std::regex;
using std::regex_search;
using boost::filesystem::path;
using boost::filesystem::recursive_directory_iterator;

vector<path> getFiles(
    path p, 
    regex pattern) {
  

  if (!exists(p) && !is_directory(p)) {
    return vector<path>();
  }
  vector<path> v = vector<path>();

  for (recursive_directory_iterator it = recursive_directory_iterator(p);
      it != recursive_directory_iterator();
      ++it) {

    if (regex_search((*it).path().filename().native(), pattern)) {
      v.push_back((*it).path());
    }

  }

  return v;
}
