// PAT32.cpp : Defines the entry point for the console application.
#include <iostream>
#include <stdio.h>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "filesystem.h"
#include "analysis.h"


using namespace std;

using boost::filesystem::is_directory;
using boost::filesystem::path;
namespace po = boost::program_options;


FILE *fp;

int main(int argc, char* argv[])
{
  path inputpath, outputpath, f, filename, s;
  string angle, s1, s2, experiment;

  po::options_description desc = po::options_description("Allowed options");
  desc.add_options()
    ("help", "print this help message")
    ("input-directory", po::value<string>(), "Input directory containing data")
    ("output-directory", po::value<string>(), "Output directory")
    ("angle", po::value<string>(), "Image angle")
    ("experiment", po::value<string>(), "Experiment to process")
  ; 

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (!(vm.count("input-directory") && 
        vm.count("output-directory") && 
        vm.count("angle") &&
        vm.count("experiment"))) {
    cout << desc << '\n';
    return 1;
  } else if (vm.count("help")){
    cout << desc << '\n'; 
    return 0;
  } else {

    inputpath = path(vm["input-directory"].as<string>());
    outputpath = path(vm["output-directory"].as<string>());
    angle = vm["angle"].as<string>();
    experiment = vm["experiment"].as<string>();

    f = path(outputpath);
    f /= "output";
    if(angle.compare("VIS_sv_000")==0)
      f += "_000.txt";
    else if(angle.compare("VIS_sv_045")==0)
      f += "_045.txt";
    else if(angle.compare("VIS_sv_090")==0)
      f += "_090.txt";

    if(is_directory(inputpath))
    {

      vector<path> file_vec = getFiles(inputpath, regex(".*\\..*"));  

      for(vector<path>::const_iterator it = file_vec.begin(); it < file_vec.end(); ++it)  
      {

        s = path(*it);

        if((s.native().find("2016-") != -1 || 
            s.native().find("2017-") != -1) && 
           s.native().find(experiment + "-") != -1 && 
           s.native().find(angle) != -1 && 
           s.native().find(".png") != -1 && 
           s.native().find(experiment + "_")!=-1)
        {
          filename = path(s);

          if(s.native().find("2016-")!=-1)
            s1=s.native().substr(s.native().find("2016-"), 10);
          if(s.native().find("2017-")!=-1)
            s1=s.native().substr(s.native().find("2017-"), 10);

          s2=s.native().substr(s.native().find(experiment + "-"), 10);

          s = path(outputpath);
          s /= s1;
          s += "_" + s2;

          if(angle.compare("VIS_sv_000")==0)
            s += "_000.jpg";
          else if(angle.compare("VIS_sv_045")==0)
            s += "_045.jpg";
          else if(angle.compare("VIS_sv_090")==0)
            s += "_090.jpg";

          cout<<filename<<"\n";

          cout << "F: " << f << '\n';
          fp=fopen(f.c_str(), "a");
          fprintf(fp, "%s,%s,", s1, s2);
          GetData(filename.c_str(), s.c_str(), s1, s2);
          fprintf(fp, "\n");
          fclose(fp);
        }

      }
    }
  }

  return 0;
}
