#include <cassert>
#include <iostream>
#include "../../csrc/quantization/marlin/sm70_marlin_common.cuh"
int main() {
  using namespace marlin::sm70;
  auto select=[](char const* f,int g,int m,int n,int k){return sm70_marlin_dense_auto_params(f,g,m,n,k);};
  for(int m : {1,2,4}) {
    auto a=select("uint4b8",32,m,2560,10240);
    auto b=select("uint4b8",32,m,10240,2560);
    assert(a.geometry.cta_n==128 && b.geometry.cta_n==128);
    assert(a.requested_split_k==1 && b.requested_split_k==1);
    assert(a.use_metadata_vector_words && b.use_metadata_vector_words);
  }
  assert(select("uint4b8",32,128,2560,10240).geometry.cta_n==256);
  assert(select("uint4b8",32,1,5120,2560).geometry.cta_n==256);
  assert(select("uint4b8",128,1,2560,10240).geometry.cta_n==256);
  assert(select("uint4",32,1,2560,10240).geometry.cta_n==256);
  std::cout<<"default-production-selector PASS\n";
}
