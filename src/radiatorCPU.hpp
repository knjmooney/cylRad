/***********************************************************************
 * Name    : radiatorCPU.cpp
 * Author  : Kevin Mooney
 * Created : 20/03/16
 * Updated : 15/08/16
 *
 * Description:
 *   Readiator class for CPU
 *
 * Notes:
 *   The begin and end member functions return the underlying
 *   1D array as opposed to the 2D array. This is purely for
 *   convenience when interacting with CUDA. The 2D array can
 *   still be accessed with the overlaoded square brackets
 *
 *   The underlying data structure is guaranteed to be contiguous
 ***********************************************************************/

#pragma once

#include <fstream>
#include <string>

#include "radiator.hpp"

class RadiatorCPU : public Radiator {
private:
  float ** _data = nullptr;

public: 
  RadiatorCPU(const size_t &nrows, const size_t &ncols);
  ~RadiatorCPU();

  typedef float * iterator;
  typedef const float * const_iterator;

  iterator       begin()        { return *_data;	  }
  iterator	 end()          { return *_data + size(); }
  const_iterator begin() const  { return *_data;	  }
  const_iterator end()   const  { return *_data + size(); }
  
  float * operator[] ( const size_t &i ) { return _data[i]; }

  // transfer policy is defined in radiator.hpp
  void copy   ( enum transferPolicy, float* begin, float* end );
  void compute(const size_t &iterations, const std::string filename = "", const size_t &skip = 1);

  // Prints to strm
  void print(std::ofstream &strm);
};

