/***********************************************************************
 * Name    : main.cpp
 * Author  : Kevin Mooney
 * Created : 20/03/16
 * Updated : 15/08/16
 *
 * Description:
 *   Main file for parsing arguments 
 *   and interacting with the radiator class
 *   
 *   The Radiator simulates heat transfusion through
 *   circular rows that are seperated by insulation
 *
 * Notes:
 ***********************************************************************/

#include <cstdio>
#include <iostream>
#include <string>
#include <unistd.h> 		// unix standard header (getopts)

#include "radiatorCPU.hpp"
#include "radiatorGPU.cuh"

using namespace std;

// Epsilon for comparing floating point values
#define EPS 1e-5

// set defaults
// these are defined global so they can be seen by getopts
string FILENAME   = "";
int    NROWS      = 32;
int    NCOLS      = 32;
int    NITERS     = 10;
int    FSKIP      = 10;      	// How many lines to skip when writing to file
bool   VERBOSE    = false;

// prints a usage message
// printf used for formatting convenience
void printUsage(const string &progName ) {
  string params[][2] = {{"-f str","file to print radiator to"      },
			{"-h    ","show this help"                 },
			{"-m int","set the number of rows"         },
			{"-n int","set the number of cols"         },
			{"-p int","set the number of iterations"   },
			{"-v    ","verbose"                        }};
  printf("Usage: ./%s [options] ...\n",progName.c_str());
  printf("Options:\n"                          );
  for ( const auto &str : params ) {
    printf("  %-25s %s\n",str[0].c_str(),str[1].c_str());
  }
}

// parse command line arguments using getopts
// make sure to update printUsage() with any new cmdline arguments
void parseArguments(int argc, char *argv[]) {
  int option;
  while(( option = getopt(argc,argv,"f:hm:n:p:v")) != -1) {
    switch(option) {
    case 'f':
      FILENAME   = optarg;
      break;
    case 'h':
      printUsage(argv[0]);
      exit(EXIT_SUCCESS);
    case 'm':
      NCOLS      = stol(optarg);
      break;
    case 'n':
      NROWS      = stol(optarg);
      break;
    case 'p':
      NITERS     = stol(optarg);
      break;
    case 'v':
      VERBOSE    = true;
      break;
    default:
      printUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }  
}

// Initialise the columns of the 
void initialiseRadiator ( RadiatorCPU & rad ) { 
  fill( rad.begin(), rad.end(), 0.0 );
  for ( size_t i=0; i < rad.nRows(); ++i ) {
    rad[i][0] = 1.00 * (float) ( i+1 ) / (float) ( rad.nRows() );
    rad[i][1] = 0.75 * (float) ( i+1 ) / (float) ( rad.nRows() );
  }
}


// Counts how many values differ by more than a predefined epsilon
size_t countDiscrepancies(float * begin1, float * end1, float * begin2) {
  size_t count = 0;
  while ( begin1 != end1 ) {
    if ( fabs ( *begin1 - *begin2 ) > EPS ) {
      count ++ ;                    
    } 
    ++begin1; ++begin2;
  }
  return count;
}

// main function
int main(int argc, char *argv[]) {
  
  parseArguments(argc,argv);
  
  if ( VERBOSE ) cout << "Initialising radiator..." << endl;
  RadiatorCPU radC(NROWS,NCOLS);
  initialiseRadiator ( radC );

  if ( VERBOSE ) cout << "Assigning memory on GPU..." << endl;
  RadiatorGPU radG(NROWS,NCOLS);

  if ( VERBOSE ) cout << "Transferring to GPU..." << endl;
  radG.copy(fromHost,radC.begin(),radC.end());

  if ( VERBOSE ) cout << "Checking Input..."      << endl;
  radG.checkInput();

  if ( VERBOSE ) cout << "Updating on CPU..." << endl;
  radC.compute(NITERS,FILENAME,FSKIP);

  if ( VERBOSE ) cout << "Updating on GPU..." << endl;
  radG.compute(NITERS);

  if ( VERBOSE ) cout << "Tranferring back to CPU..." << endl;
  RadiatorCPU resultGPU(NROWS,NCOLS);
  resultGPU.copy(fromDevice,radG.begin(),radG.end());

  if ( VERBOSE ) {
    size_t discrep = countDiscrepancies(radC.begin(),radC.end(),resultGPU.begin());
    cout << "\nThere are "<< discrep << " discrepancies greater than " << EPS << endl;
  }
}
