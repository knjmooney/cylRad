/*********************************************************
 * Name   : radiator.hpp
 * Author : Kevin Mooney
 * Date   : 22/03/16
 *
 * Abstract Radiator Base class
 *
 * NOTE: 
 * and solution to GPU gets stored in nextMatrix
 *********************************************************/

#pragma once

class Radiator {

protected:
  const size_t _nRows;
  const size_t _nCols;
  const size_t _size;

public:
  // constructor for an m*n radiator matrix
  Radiator(const size_t &m, const size_t &n) : _nRows{m}, _nCols{n}, _size{m*n} {}
  ~Radiator() {}

  size_t nRows() const { return _nRows; }
  size_t nCols() const { return _nCols; }
  
  // Returns nRows*nCols
  size_t size()  const { return _size; }
};

// Tranfer Policies for copy member functions
enum  transferPolicy {fromHost,fromDevice};
