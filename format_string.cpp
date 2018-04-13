#include "format_string.h"

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>

using std::string;

string stringf(string format, ...) {

  va_list args;
  int size;
  string output;

  va_start(args, format);
  size = vsnprintf(nullptr, 0, format.c_str(), args);
  va_end(args);

  if (size < 0) {
    // Dont know what to do now
    throw "Encoding error in vsnprintf";
  }

  char c_str_out[size];

  va_start(args, format);
  vsnprintf(c_str_out, size, format.c_str(), args);
  va_end(args);

  output = string(c_str_out);
  return output;
}

