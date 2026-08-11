#pragma once

#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace nerd {

/*
 * Shape class creates vector of size_t numbers to define the dimensions or "shape" of a tensor
 */
class Shape {
  private:
    std::vector<std::size_t> dim;

  public:
    Shape() = default;
    Shape(std::initializer_list<std::size_t> dim) : dim(dim) {}
    Shape(std::size_t rank) : dim(rank, 1) {}

    // Access operators
    std::size_t& operator[](std::size_t i) {
      assert(i < dim.size());
      return dim[i];
    }

    const std::size_t& operator[](std::size_t i) const {
      assert(i < dim.size());
      return dim[i];
    }

    // overload std::cout so displaying shape is possible
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
      os << "{";
      for (size_t i = 0; i < shape.rank(); ++i) {
          os << shape[i];
          if (i + 1 != shape.rank()) {
              os << ", ";
          }
      }
      os << "}";
      return os;
    }

    /*
     * Iterators
     */
    auto begin() { return dim.begin(); }
  
    auto end() { return dim.end(); }
    
    auto begin() const { return dim.begin(); }

    auto end() const { return dim.end(); }

    // returns number of dimensions of tensor
    std::size_t rank() const {
      return dim.size();
    }

    // another way to say rank but not for linear algebra nerds - Aidan Boyce (linear algebra nerd)
    std::size_t ndim() const {
      return rank();
    }
    
    // removes inputted value from shape object (may soon be deprecated)
    void remove_val(std::size_t val) {
      dim.erase(std::remove(dim.begin(), dim.end(), val), dim.end());
    }

    // removes value at index of shape object
    void remove_dim(std::size_t index) {
      assert(index < dim.size());
      dim.erase(dim.begin() + index);
    }

    // adds inputted value at index of shape object
    void add(std::size_t index, std::size_t val) {
      dim.insert(dim.begin() + index, val);
    }

    // return total number of elements in tensor
    std::size_t size() const {
      if (dim.empty()) return 0;
      std::size_t total = 1;
      for (std::size_t d : dim) {
        total *= d;
      }
      return total;
    }

    // operator to checks if shapes are equal
    bool operator==(const Shape& other) const {
        return dim == other.dim;
    }

    // opposite of operator above
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    // checks if tensor is a matrix
    bool is_matrix() const {
      return dim.size() == 2;
    }

    // checks if tensor is square matrix
    bool is_square() const {
      return dim[0] == dim[1] && is_matrix();
    }

    // checks if shape has any values inside of it
    bool empty() const {
      return dim.empty();
    }
};

}
