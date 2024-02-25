cmake_minimum_required(VERSION 3.12)
project(aip-hwfin LANGUAGES CUDA CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust the architecture according to your needs

cmake_policy(SET CMP0148 OLD)
cmake_policy(SET CMP0146 OLD)
add_subdirectory(pybind11)

# binding for Tensor
pybind11_add_module(myTensor src/bind_Tensor.cu) # all cpp files

target_link_libraries(myTensor PRIVATE ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(myTensor PRIVATE "/home/ou524u/anaconda3/envs/nerf-studio/lib/libcurand.so")
target_link_libraries(myTensor PRIVATE "/home/ou524u/anaconda3/envs/nerf-studio/lib/libcublas.so")

# # binding for Layer
pybind11_add_module(myLayer src/bind_Layer.cu) # all cpp files

target_link_libraries(myLayer PRIVATE ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(myLayer PRIVATE "/home/ou524u/anaconda3/envs/nerf-studio/lib/libcurand.so")
target_link_libraries(myLayer PRIVATE "/home/ou524u/anaconda3/envs/nerf-studio/lib/libcublas.so")

# target_link_libraries(myLayer PRIVATE ${CUDA_LIBRARIES} -lcublas -lcurand)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# cmake -D CUDACXX=/mnt/d/Cuda11.8 ..