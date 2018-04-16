// PAT32.cpp : Defines the entry point for the console application.
#include <iostream>
#include <stdio.h>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "filesystem.h"
#include "analysis.h"
#include "format_string.h"


using namespace std;

using boost::filesystem::exists;
using boost::filesystem::path;
namespace po = boost::program_options;


int main(int argc, char* argv[])
{
  path infile, outfile;
  string date, plant_id;

  po::options_description desc = po::options_description("Allowed options");
  desc.add_options()
    ("help", "print this help message")
    ("input-file", po::value<string>(), "Input image file")
    ("date", po::value<string>(), "Date of image acquisition")
    ("plant-id", po::value<string>(), "Plant ID")
    ("output-file", po::value<string>(), "Masked output image (jpeg)")
    ; 

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (!vm.count("input-file")) {
    cerr << desc << endl;
    return 1;
  } else if (vm.count("help")){
    cout << desc << endl; 
    return 0;
  } 

  infile = path(vm["input-file"].as<string>());

  if (vm.count("date"))
    date = vm["date"].as<string>();
  else
    date = "NA";

  if (vm.count("plant-id"))
    plant_id = vm["plant-id"].as<string>();
  else
    plant_id = "NA";

  if(!exists(infile)) {
    cerr << "File not found: " << infile.native() << endl;
    return 1;
  }

  if (vm.count("output-file")) {
    outfile = path(vm["output-file"].as<string>());

    if (!is_directory(outfile.parent_path())) {
      cerr << "Cannot create file: " + outfile.native() +
        "\n Parent directory does not exist." << endl;
      return 1;
    }
  }



  struct plant_data p_data = GetData(infile.native().c_str());


  std::string sep = ", ";
  cout << plant_id << sep <<
    date << sep <<
    p_data.to_string(sep) << endl;

  if (vm.count("output-file"))
    cv::imwrite(outfile.native(), p_data.image);

  return 0;
}
