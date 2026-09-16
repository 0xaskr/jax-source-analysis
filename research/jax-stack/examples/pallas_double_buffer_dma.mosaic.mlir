module attributes {stable_mosaic.version = 15 : i64} {
  func.func @manual_double_buffer_add(%arg0: memref<40x128xf32, #tpu.memory_space<hbm>>, %arg1: memref<40x128xf32, #tpu.memory_space<hbm>>, %arg2: memref<40x128xf32, #tpu.memory_space<hbm>>, %arg3: memref<2x8x128xf32, #tpu.memory_space<vmem>>, %arg4: memref<2x8x128xf32, #tpu.memory_space<vmem>>, %arg5: memref<2x8x128xf32, #tpu.memory_space<vmem>>, %arg6: memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>, %arg7: memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>, %arg8: memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 6 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.remsi %c0_i32, %c2_i32 : i32
    %1 = tpu.memref_slice %arg6[%0] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %2 = tpu.memref_squeeze %1 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %3 = tpu.memref_slice %arg3[%0, %c0_i32_0, %c0_i32_1] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
    %4 = tpu.memref_squeeze %3 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %5 = tpu.memref_slice %arg0[%c0_i32_2, %c0_i32_3] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    tpu.enqueue_dma source(%5 : memref<8x128xf32, #tpu.memory_space<hbm>>) target(%4 : memref<8x128xf32, #tpu.memory_space<vmem>>) target_semaphore(%2 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    %c0_i32_4 = arith.constant 0 : i32
    %c2_i32_5 = arith.constant 2 : i32
    %6 = arith.remsi %c0_i32_4, %c2_i32_5 : i32
    %7 = tpu.memref_slice %arg7[%6] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %8 = tpu.memref_squeeze %7 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %9 = tpu.memref_slice %arg4[%6, %c0_i32_6, %c0_i32_7] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
    %10 = tpu.memref_squeeze %9 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %c0_i32_8 = arith.constant 0 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %11 = tpu.memref_slice %arg1[%c0_i32_8, %c0_i32_9] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    tpu.enqueue_dma source(%11 : memref<8x128xf32, #tpu.memory_space<hbm>>) target(%10 : memref<8x128xf32, #tpu.memory_space<vmem>>) target_semaphore(%8 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    %c0_i32_10 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %12 = arith.addi %c0_i32_10, %c5_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg9 = %c0_i32_10 to %12 step %c1_i32  : i32 {
      %c2_i32_20 = arith.constant 2 : i32
      %25 = arith.remsi %arg9, %c2_i32_20 : i32
      %c2_i32_21 = arith.constant 2 : i32
      %26 = arith.remsi %arg9, %c2_i32_21 : i32
      %c8_i32 = arith.constant 8 : i32
      %27 = arith.muli %arg9, %c8_i32 : i32
      %28 = tpu.memref_slice %arg6[%26] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %29 = tpu.memref_squeeze %28 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %c0_i32_22 = arith.constant 0 : i32
      %c0_i32_23 = arith.constant 0 : i32
      %30 = tpu.memref_slice %arg3[%26, %c0_i32_22, %c0_i32_23] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
      %31 = tpu.memref_squeeze %30 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
      %c0_i32_24 = arith.constant 0 : i32
      %32 = tpu.memref_slice %arg0[%27, %c0_i32_24] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
      tpu.wait_dma2 semaphore(%29 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%32 : memref<8x128xf32, #tpu.memory_space<hbm>>) dst(%31 : memref<8x128xf32, #tpu.memory_space<vmem>>)
      %c2_i32_25 = arith.constant 2 : i32
      %33 = arith.remsi %arg9, %c2_i32_25 : i32
      %c8_i32_26 = arith.constant 8 : i32
      %34 = arith.muli %arg9, %c8_i32_26 : i32
      %35 = tpu.memref_slice %arg7[%33] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %36 = tpu.memref_squeeze %35 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %c0_i32_27 = arith.constant 0 : i32
      %c0_i32_28 = arith.constant 0 : i32
      %37 = tpu.memref_slice %arg4[%33, %c0_i32_27, %c0_i32_28] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
      %38 = tpu.memref_squeeze %37 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
      %c0_i32_29 = arith.constant 0 : i32
      %39 = tpu.memref_slice %arg1[%34, %c0_i32_29] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
      tpu.wait_dma2 semaphore(%36 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%39 : memref<8x128xf32, #tpu.memory_space<hbm>>) dst(%38 : memref<8x128xf32, #tpu.memory_space<vmem>>)
      %c1_i32_30 = arith.constant 1 : i32
      %40 = arith.addi %arg9, %c1_i32_30 : i32
      %c5_i32_31 = arith.constant 5 : i32
      %41 = arith.cmpi slt, %40, %c5_i32_31 : i32
      %42 = arith.extui %41 : i1 to i32
      %c0_i32_32 = arith.constant 0 : i32
      %43 = arith.cmpi ne, %42, %c0_i32_32 : i32
      scf.if %43 {
        %c1_i32_45 = arith.constant 1 : i32
        %65 = arith.addi %arg9, %c1_i32_45 : i32
        %c2_i32_46 = arith.constant 2 : i32
        %66 = arith.remsi %65, %c2_i32_46 : i32
        %c8_i32_47 = arith.constant 8 : i32
        %67 = arith.muli %65, %c8_i32_47 : i32
        %68 = tpu.memref_slice %arg6[%66] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %69 = tpu.memref_squeeze %68 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %c0_i32_48 = arith.constant 0 : i32
        %c0_i32_49 = arith.constant 0 : i32
        %70 = tpu.memref_slice %arg3[%66, %c0_i32_48, %c0_i32_49] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
        %71 = tpu.memref_squeeze %70 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
        %c0_i32_50 = arith.constant 0 : i32
        %72 = tpu.memref_slice %arg0[%67, %c0_i32_50] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
        tpu.enqueue_dma source(%72 : memref<8x128xf32, #tpu.memory_space<hbm>>) target(%71 : memref<8x128xf32, #tpu.memory_space<vmem>>) target_semaphore(%69 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
        %c1_i32_51 = arith.constant 1 : i32
        %73 = arith.addi %arg9, %c1_i32_51 : i32
        %c2_i32_52 = arith.constant 2 : i32
        %74 = arith.remsi %73, %c2_i32_52 : i32
        %c8_i32_53 = arith.constant 8 : i32
        %75 = arith.muli %73, %c8_i32_53 : i32
        %76 = tpu.memref_slice %arg7[%74] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %77 = tpu.memref_squeeze %76 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %c0_i32_54 = arith.constant 0 : i32
        %c0_i32_55 = arith.constant 0 : i32
        %78 = tpu.memref_slice %arg4[%74, %c0_i32_54, %c0_i32_55] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
        %79 = tpu.memref_squeeze %78 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
        %c0_i32_56 = arith.constant 0 : i32
        %80 = tpu.memref_slice %arg1[%75, %c0_i32_56] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
        tpu.enqueue_dma source(%80 : memref<8x128xf32, #tpu.memory_space<hbm>>) target(%79 : memref<8x128xf32, #tpu.memory_space<vmem>>) target_semaphore(%77 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
      } else {
      }
      %c2_i32_33 = arith.constant 2 : i32
      %44 = arith.cmpi sge, %arg9, %c2_i32_33 : i32
      %45 = arith.extui %44 : i1 to i32
      %c0_i32_34 = arith.constant 0 : i32
      %46 = arith.cmpi ne, %45, %c0_i32_34 : i32
      scf.if %46 {
        %c2_i32_45 = arith.constant 2 : i32
        %65 = arith.subi %arg9, %c2_i32_45 : i32
        %c2_i32_46 = arith.constant 2 : i32
        %66 = arith.remsi %65, %c2_i32_46 : i32
        %c8_i32_47 = arith.constant 8 : i32
        %67 = arith.muli %65, %c8_i32_47 : i32
        %68 = tpu.memref_slice %arg8[%66] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %69 = tpu.memref_squeeze %68 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %c0_i32_48 = arith.constant 0 : i32
        %70 = tpu.memref_slice %arg2[%67, %c0_i32_48] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
        %c0_i32_49 = arith.constant 0 : i32
        %c0_i32_50 = arith.constant 0 : i32
        %71 = tpu.memref_slice %arg5[%66, %c0_i32_49, %c0_i32_50] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
        %72 = tpu.memref_squeeze %71 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
        tpu.wait_dma2 semaphore(%69 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%72 : memref<8x128xf32, #tpu.memory_space<vmem>>) dst(%70 : memref<8x128xf32, #tpu.memory_space<hbm>>)
      } else {
      }
      %47 = arith.index_cast %25 : i32 to index
      %c0 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %48 = vector.load %arg3[%47, %c0, %c0_35] : memref<2x8x128xf32, #tpu.memory_space<vmem>>, vector<1x8x128xf32>
      %49 = vector.shape_cast %48 : vector<1x8x128xf32> to vector<8x128xf32>
      %50 = arith.index_cast %25 : i32 to index
      %c0_36 = arith.constant 0 : index
      %c0_37 = arith.constant 0 : index
      %51 = vector.load %arg4[%50, %c0_36, %c0_37] : memref<2x8x128xf32, #tpu.memory_space<vmem>>, vector<1x8x128xf32>
      %52 = vector.shape_cast %51 : vector<1x8x128xf32> to vector<8x128xf32>
      %53 = arith.addf %49, %52 : vector<8x128xf32>
      %54 = arith.index_cast %25 : i32 to index
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %55 = vector.load %arg5[%54, %c0_38, %c0_39] : memref<2x8x128xf32, #tpu.memory_space<vmem>>, vector<1x8x128xf32>
      %56 = vector.shape_cast %55 : vector<1x8x128xf32> to vector<8x128xf32>
      %57 = vector.shape_cast %53 : vector<8x128xf32> to vector<1x8x128xf32>
      tpu.vector_store %arg5[%54, %c0_38, %c0_39], %57 {strides = array<i32>} : memref<2x8x128xf32, #tpu.memory_space<vmem>>, vector<1x8x128xf32>, 
      %c2_i32_40 = arith.constant 2 : i32
      %58 = arith.remsi %arg9, %c2_i32_40 : i32
      %c8_i32_41 = arith.constant 8 : i32
      %59 = arith.muli %arg9, %c8_i32_41 : i32
      %60 = tpu.memref_slice %arg8[%58] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %61 = tpu.memref_squeeze %60 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
      %c0_i32_42 = arith.constant 0 : i32
      %62 = tpu.memref_slice %arg2[%59, %c0_i32_42] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
      %c0_i32_43 = arith.constant 0 : i32
      %c0_i32_44 = arith.constant 0 : i32
      %63 = tpu.memref_slice %arg5[%58, %c0_i32_43, %c0_i32_44] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
      %64 = tpu.memref_squeeze %63 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
      tpu.enqueue_dma source(%64 : memref<8x128xf32, #tpu.memory_space<vmem>>) target(%62 : memref<8x128xf32, #tpu.memory_space<hbm>>) target_semaphore(%61 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    }
    %c5_i32_11 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32_12 = arith.constant 2 : i32
    %13 = arith.remsi %c3_i32, %c2_i32_12 : i32
    %14 = tpu.memref_slice %arg8[%13] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %15 = tpu.memref_squeeze %14 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %c24_i32 = arith.constant 24 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %16 = tpu.memref_slice %arg2[%c24_i32, %c0_i32_13] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    %c0_i32_14 = arith.constant 0 : i32
    %c0_i32_15 = arith.constant 0 : i32
    %17 = tpu.memref_slice %arg5[%13, %c0_i32_14, %c0_i32_15] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
    %18 = tpu.memref_squeeze %17 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    tpu.wait_dma2 semaphore(%15 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%18 : memref<8x128xf32, #tpu.memory_space<vmem>>) dst(%16 : memref<8x128xf32, #tpu.memory_space<hbm>>)
    %c4_i32 = arith.constant 4 : i32
    %c2_i32_16 = arith.constant 2 : i32
    %19 = arith.remsi %c4_i32, %c2_i32_16 : i32
    %20 = tpu.memref_slice %arg8[%19] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %21 = tpu.memref_squeeze %20 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %c32_i32 = arith.constant 32 : i32
    %c0_i32_17 = arith.constant 0 : i32
    %22 = tpu.memref_slice %arg2[%c32_i32, %c0_i32_17] : memref<40x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    %c0_i32_18 = arith.constant 0 : i32
    %c0_i32_19 = arith.constant 0 : i32
    %23 = tpu.memref_slice %arg5[%19, %c0_i32_18, %c0_i32_19] : memref<2x8x128xf32, #tpu.memory_space<vmem>> -> memref<1x8x128xf32, #tpu.memory_space<vmem>>
    %24 = tpu.memref_squeeze %23 : memref<1x8x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    tpu.wait_dma2 semaphore(%21 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%24 : memref<8x128xf32, #tpu.memory_space<vmem>>) dst(%22 : memref<8x128xf32, #tpu.memory_space<hbm>>)
    return
  }
}

