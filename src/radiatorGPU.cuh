/***********************************************************************
 * Name    : radiatorGPU.cpp
 * Author  : Kevin Mooney
 * Created : 20/03/16
 * Updated : 11/08/16
 *
 * Description:
 *   Radiator class for GPU
 *
 * Notes:
 ***********************************************************************/

#pragma once

#include "radiator.hpp"

class RadiatorGPU : public Radiator {
private:
  float *   _data = nullptr;	// device pointer

public:
  // Mallocs a device pointer
  RadiatorGPU(const size_t &nrows, const size_t &ncols);
  // Destroys device pointer
  ~RadiatorGPU();

  // Iterators are device pointers
  typedef float * iterator;
  typedef const float * const_iterator;
  iterator begin() { return _data; }
  iterator end()   { return _data + size(); }
  const_iterator begin() const { return _data; }
  const_iterator end()   const { return _data + size(); }

  // Copies with transfer policy defined in radiator.hpp
  void copy ( enum transferPolicy, float* begin, float* end );

  // Update the matrix iterations number of times
  void compute(const size_t &iterations);

  // Checks integrety of input
  void checkInput();
};
