module attributes {stable_mosaic.version = 15 : i64} {
  func.func @pallas_add(%arg0: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg1: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg2: memref<8x128xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0_0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %1 = vector.load %arg1[%c0_1, %c0_2] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %2 = arith.addf %0, %1 : vector<8x128xf32>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %3 = vector.load %arg2[%c0_3, %c0_4] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    tpu.vector_store %arg2[%c0_3, %c0_4], %2 {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>, 
    return
  }
}

