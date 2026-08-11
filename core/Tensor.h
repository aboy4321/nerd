#pragma once

#include <Shape.h>
#include <vector>
#include <cassert>
#include <cstddef> 
#include <iostream>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cmath>

namespace nerd {

template <typename Type>
class Tensor {
  private:

    /*
     * shape - dimensions the tensor takes on 
     * data - self explanatory
     * strides - helps w/ indexing in the tensor the inner workings of it are funky
     */
    Shape shape;
    std::vector<Type> data;
    Shape strides;    

    // computing strides to help with multidim indexing
    Shape compute_strides() {
      Shape strides(shape.rank());
      for (int i = shape.rank() - 2; i >= 0; --i ) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
      return strides;
    }


    // helper printing function, takes into account varying dimensions, thus nesting bracketss where necessary
    void print_recursive(
        std::ostream& os,
        std::size_t dim,
        std::size_t offset,
        std::size_t indent
    ) const
    {
        std::string spacing(indent, ' ');
        os << spacing << "[\n";
        // Last dimension: print values
        if (dim == shape.rank() - 1)
        {
            os << spacing << " ";
            for (std::size_t i = 0; i < shape[dim]; ++i)
            {
                os << data[offset + i * strides[dim]];
                if (i + 1 < shape[dim])
                    os << ", ";
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape[dim]; ++i)
            {
                print_recursive(
                    os,
                    dim + 1,
                    offset + i * strides[dim],
                    indent + 2
                );
                if (i + 1 < shape[dim])
                    os << ",\n";
            }
        }
        os << "\n" << spacing << "]";
    }

    /*
     * Given vector of indices, converts into flat data index via strides
     */
    std::size_t flat_index(const std::vector<std::size_t>& indices) const {
      assert(indices.size() == shape.rank());

      std::size_t index = 0;
      for (std::size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape[i]);
        index += indices[i] * strides[i];
      }
      return index;
    }
  public:

    /* 
     * Constructors and factory functions
     */

    // empty constructor
    Tensor() = default;

    // takes in shape, data is initialized as zeros
    Tensor(const Shape& shape)
      : shape(shape), data(shape.size()) {
        strides = compute_strides();
      }

    // takes and shape, makes data some uniform inputted value 
    Tensor(const Shape& shape, const Type& val)
      : shape(shape), data(shape.size(), val) {
        strides = compute_strides();
      } 
  
    // returns tensor of 0s
    static Tensor zeros(const Shape& shape) {
      return Tensor(shape, Type{0});
    }

    // returns tensors of 1s
    static Tensor ones(const Shape& shape) {
      return Tensor(shape, Type{1});
    }

    // returns identity matrix
    static Tensor identity(std::size_t dim) {
      Tensor<Type> res(Shape{dim, dim});
      for (std::size_t i = 0; i < dim; ++i) {
        res.data[i * dim + i] = Type{1};
      }
      return res;
    }

    // returns diagonal matrix of inputs
    static Tensor diagonal(const std::vector<Type>& arr) {
      std::size_t dim = arr.size();
      Tensor<Type> res(Shape{dim, dim});
      for (std::size_t i = 0; i < dim; ++i) {
        res.data[i * dim + i] = arr[i];
      }
      return res;
    }

    /*
     * Accesing data and data at index + printing
     */

    /*
     * Performs opposite operation of flat_index, converts flat index into vector of indices
     */
    std::vector<std::size_t> unravel(std::size_t index) const {
      std::vector<std::size_t> coords(shape.rank());
      for (std::size_t i = 0; i < shape.rank(); ++i) {
        coords[i] = index / strides[i];
        index %= strides[i];
      }
      return coords;
    }

    // For modification of data 
    Type& operator[](std::size_t i) {
      assert(i < data.size());
      return data[i];
    }

    // For reading data
    const Type& operator[](std::size_t i) const {
      assert(i < data.size());
      return data[i];
    }

    // For modification w/ multidimensional indexing
    Type& operator()(const std::vector<std::size_t>& indices) {
      return data[flat_index(indices)];
    }

    // Read only version of operator above 
    const Type& operator()(const std::vector<std::size_t>& indices) const {
      return data[flat_index(indices)];
    }    

    Type& operator()(std::size_t r, std::size_t c) {
      return data[r * strides[0] + c * strides[1]];
    }

    const Type& operator()(std::size_t r, std::size_t c) const {
      return data[r * strides[0] + c * strides[1]];
    }

    // Another helper print thingy
    void print(std::ostream& os) const {
        print_recursive(os, 0, 0, 0);
    }

    // NOTE -- DO NOT USE THIS ABOVE 3-4 DIMENSIONS IT IS PRACTICALLY UNREADABLE
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
      tensor.print(os);
      return os;
    }

    /* 
     * Tensor arithmetic
     */

    // Modification by addition
    Tensor& operator+=(const Tensor& other) {
      assert(same_shape(other));
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
      }
      return *this;
    }

    // Creates another tensor via addition
    Tensor operator+(const Tensor& other) const {
      Tensor res = *this;
      res += other;
      return res;
    }

    // Modification by subtraction
    Tensor& operator-=(const Tensor& other) {
      assert(same_shape(other));
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
      }
      return *this;
    }

    // Creates another tensor via subtraction
    Tensor operator-(const Tensor& other) const {
      Tensor res = *this;
      res -= other;
      return res;
    }

    // Modifies tensor via scalar multiplication
    Tensor& operator*=(const Type& n) {
      for (Type& x : data) {
        x *= n;
      }
      return *this;
    }

    // Creates another tensor via scalar multiplication
    Tensor operator*(const Type& n) const {
      Tensor res = *this;
      res *= n;
      return res;
    }

    // and below is scalar division
    Tensor& operator/=(const Type& n) {
      for (Type& x : data) {
        x /= n;
      }
      return *this;
    }

    Tensor operator/(const Type& n) const {
      Tensor res = *this;
      res /= n;
      return res;
    }

    /*
     * Generic Tensor Manipulation
     */

    /*
     * Function will soon be deprecated ignore
     */
    Tensor m_transpose() const {
      assert(ndim() == 2);
      Tensor res(Shape{shape[1], shape[0]});
      for (std::size_t i = 0; i < shape[1]; ++i) {
        for (std::size_t j = 0; j < shape[0]; ++j) {
          res({i, j}) = (*this)({j, i});
        }
      }
      return res;
    }

    // reshapes tensor, data is not affected
    Tensor reshape(const Shape& other) const {
      assert(shape.size() == other.size());
      Tensor res = *this;
      res.shape = other;
      res.strides = res.compute_strides();
      return res;
    }

    // "flattens" tensor into 1 x size() array
    Tensor flatten() const {
      return reshape({size()});
    }

    // removes dimensions of 1 inside tensor
    Tensor squeeze() const {
      Tensor res = *this;
      res.shape.remove_val(1);
      res.strides = res.compute_strides();
      return res;
    }

    // adds dimensions of 1 inside tensor at inputted index
    Tensor unsqueeze(std::size_t index) const {
      Tensor res = *this;
      res.shape.add(index, 1);
      res.strides = res.compute_strides();
      return res;
    }

    // given the shape of previous tensor object, input permutation of indices to change OG tensor shape 
    Tensor permute(const std::vector<std::size_t>& other) const {
      assert(shape.rank() == other.size());
      Tensor res(*this);
      for (std::size_t i = 0; i < other.size(); ++i) {
        res.shape[i] = this->shape[other[i]];
        res.strides[i] = this->strides[other[i]];
      }
      return res;
    }

    /*
     *  Iterators
     */

    auto begin() {
      return data.begin();
    }

    auto end() {
      return data.end();
    }

    auto begin() const {
      return data.begin();
    }

    auto end() const {
      return data.end();
    }

    // Fills tensor w/ inputted value
    void fill(const Type& T) {
      for (auto& x : data) {
        x = T;
      }
    }

    // Checks if a tensor has elements
    bool empty() const {
      return data.empty();
    }

    // Checks if a tensor has the same shape as another tensor
    bool same_shape(const Tensor& other) const {
      return shape == other.shape;
    }

    /* 
     * Getter functions
     */ 

    // Returns shape of tensor
    const Shape& get_shape() const {
      return shape;
    }

    const std::size_t get_shape(std::size_t index) const {
      return shape[index];
    }

    // Returns strides of a tensors dimensions
    const Shape& get_strides() const {
      return strides;
    }

    // Returns number of dimensions tensor has
    std::size_t ndim() const {
      return shape.ndim();
    }

    // Returns size of tensor/number of elements 
    std::size_t size() const {
      return shape.size();
    }
};
  
}
