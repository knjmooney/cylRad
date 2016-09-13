/*********************************************************
 * Name    : radiatorCPU.cpp
 * Author  : Kevin Mooney
 * Created : 22/03/16
 * Updated : 15/08/16
 *
 * Description:
 *   Definitions of the RadiatorCPU class
 *
 * Notes:
 *********************************************************/

#include <fstream>
#include <stdexcept>
#include <string>

#include "radiatorCPU.hpp"
#include "kernelInteractions.cuh"

// constructer
// remember to call delete in destructor
RadiatorCPU::RadiatorCPU(const size_t &m, const size_t &n) : Radiator{m,n} {
  _data            = new float*[_nRows];
  float * onedp    = new float [_nRows*_nCols]; 
  // set 2D pointers to the start of each row
  for ( size_t i=0; i<_nRows; i++ ) {
    _data[i] = onedp + _nCols*i;
  }
}

// delete all news called in constructor
RadiatorCPU::~RadiatorCPU() {
  delete[] _data[0];
  delete[] _data;
}

// Helper function for updating the matrix
// Could possibly make this a friend of the class rather than having to pass all the data
void update (float ** data, float ** oldData, const size_t &nRows, const size_t &nCols ) {
  for ( size_t ui=0; ui<nRows; ui++ ) {
    for ( size_t uj=2; uj<nCols; uj++ ) {
      
      data [ ui ][ uj ] = ( + ( 0.37 * oldData [ ui ][ uj-2           ] )
			    + ( 0.28 * oldData [ ui ][ uj-1           ] )
			    + ( 0.20 * oldData [ ui ][ uj             ] )
			    + ( 0.12 * oldData [ ui ][ (uj+1) % nCols ] )
			    + ( 0.03 * oldData [ ui ][ (uj+2) % nCols ] ) 
			    );
    }
  }
}

// update the matrix nIters number of times
void RadiatorCPU::compute(const size_t &nIters, const std::string filename, const size_t & skip) {
  // Initialise 2D matrix for storing the old answer
  float **oldData      = new float*[_nRows];
  float * onedp        = new float [_nRows*_nCols]();
  for ( size_t i=0; i<_nRows; i++ ) {
    oldData[i] = onedp + _nCols*i;
  }
  std::copy ( begin(), end(), onedp ); 
  
  if ( filename == "" ) {	// Don't output matrix to file
    for ( size_t i=0; i<nIters; i++ ) {
      update(_data,oldData,_nRows,_nCols);
      std::swap(_data,oldData);
    }
  }
  else {
    std::ofstream strm (filename,std::ofstream::out);
    for ( size_t i=0; i<nIters; i++ ) {
      update(_data,oldData,_nRows,_nCols);
      std::swap(_data,oldData);
      if ( i%skip == 0 ) print(strm);
    }    
  }

  // Delete the locally defined 2D matrix
  delete[] oldData;
  delete[] onedp;
}

// copies relative to transferPolicies defined in radiator.hpp
void RadiatorCPU::copy ( enum transferPolicy tp, float* begin, float* end ) {
  switch ( tp ) {
  case fromHost:
    std::copy ( begin, end, *_data ) ;
    break;
  case fromDevice:
    transfer2Host ( begin, end, *_data );
    break;
  default:			// Unknown Policy
    throw (std::invalid_argument("Unrecognised transfer policy") );
  }
}

// Outputs matrix onto strm
void RadiatorCPU::print(std::ofstream &strm) {
  for ( size_t i=0; i<_nCols; i++ ) {
    for ( size_t j=0; j<_nRows; j++ ) {
      strm <<_data[i][j] << " ";
    }
    strm << std::endl;
  }
  strm << std::endl << std::endl;
}

