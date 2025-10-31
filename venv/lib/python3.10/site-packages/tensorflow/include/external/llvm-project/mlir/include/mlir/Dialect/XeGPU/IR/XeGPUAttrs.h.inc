/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace xegpu {
/// a composite attribute for `TensorDescType`
/// `BlockTensorDesc` (or `block_tdesc_attr`) is a composite
///     attribute defined for `TensorDescType` for describing following
///     properties of a `TensorDesc`.
///     1. `memory_space`: It describes where the data block described by the
///         TensorDesc is located, `Global` device memory or `Shared` local memory.
///         It is default to `Global`.
///     2. `array_length`: It describes how many horizontally consecutive blocks
///         will be loaded by a hardware load instruction. If the TensorDesc shape
///         is 8x16, with array_length = 2. The loaded block shape will be actually
///         8x32. Its default value is 1.
///     3. `boundary_check`: It is used to indicates the hardware whether to do
///         out-of-boundary check. The default value is true.
class BlockTensorDescAttr;
/// a composite attribute for `TensorDescType`
/// `ScatterTensorDesc` is a composite attribute defined for `TensorDescType`
///     for describing following properties of a `TensorDesc`:
/// 
///     1. `memory_space`: It describes where the data block described by the
///         TensorDesc is located, `Global` device memory or `Shared` local memory.
///         It is default to `Global`.
/// 
///     2.  `chunk_size`: indicates number of contiguous elements accessed for each
///         offset, default is 1. It is used with `scattered` attr only.
class ScatterTensorDescAttr;
/// Describe the location of data described by a `TensorDesc`:
///                  Global device memory (`Global`) or Shared local memory (`SLM`).
class MemorySpaceAttr;
/// Describe the cache settings for prefetch/load/store operators
class CachePolicyAttr;
/// Describes the scope of fence.
///                     "workgroup" means that the scope is within each work group.
///                     "gpu" means the scope is across work groups within the gpu.
class FenceScopeAttr;
/// Describes the data distribution to subgroups and work-items for a tensor
///     specified by the tensor descriptor.
/// XeGPU operations use `LayoutAttr` to define how data is distributed across subgroups and work-items.
///     This attribute is specified in tensor descriptors during tensor description creation. `LayoutAttr`
///     includes the following parameters:
/// 
///     * `sg_layout`: Specifies the total number of subgroups and their layout within a workgroup.
///       It is mandatory for workgroup-level programming. Its presence implies workgroup-level code.
///     * `sg_data`: Defines the data size accessed per subgroup. It is optionally used with `sg_layout`
///       for workgroup-level programming. When it is left empty, the size accessed per subgroup can be
///       derived from the tensor shape and `sg_layout` using the formula:
///       `sg_data[i] = tensor_shape[i] / sg_layout[i]`.
///     * `inst_data`: Specifies the data size that is processed by an instruction. It is optionally
///       used with lane_layout. When it is left empty, the data size per instruction is equivalent to
///       the sg_data for workgroup-level programming or equivalent to tensor shape for subgroup-level
///       programming.
///     * `lane_layout` : Specifies the total number of work-items and their arrangement within a subgroup.
///       It is mandatory for subgroup-level programming and optional for workgroup-level programming.
///     * `lane_data` : Specifies the shape of the tensor fragment that each lane accesses. It defines a single,
///       minimal distribution unit. Processing the entire tensor may require one or more distribution units per
///       hardware instruction.
///     * `order`: Specifies the dimension order used to linearize n-dimensional sg_layout and lane_layout to
///       1-dimensional layout. The first dimension in the order list is the fastest-changing dimension. If it
///       is not present, the default value is [1, 0].
/// 
///     Examples:
/// 
///     1. Subgroup level layout:
///     ```mlir
///     #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     In this example, there are 16 work-items per subgroup, and is organized as
///     [[0, 1, 2, .., 7],[8, 9, .., 15]]. The distribution unit is 1x1.
/// 
///     2. Subgroup level layout with order:
///     ```mlir
///     #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
///     ```
///     In this example, there are 16 work-items per subgroup, and is organized as
///     [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]]. The distribution unit is 1x1.
/// 
///     3. Subgroup level layout with inst_data
///     ```mlir
///     #xegpu.layout<inst_data = [8, 16], lane_layout = [2, 8], lane_data = [2, 2]>
///     ```
///     In this example, the original problem size is partitioned into smaller subproblems of dimensions [8, 16],
///     which are then distributed among 16 work-items arranged as [[0, 1, 2, ..., 7], [8, 9, ..., 15]]. Each
///     work-item is assigned four 2x2 blocks in a round-robin manner.
/// 
///     4. Workgroup level layout:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
///     arranged as [[0, 1, 2, 3], [4, 5, 6, 7]]. Each subgroup accesses a 16x16 block per instruction, which
///     is further distributed to 16 work items which is organized as [[0, 1, 2, .., 7],[8, 9, .., 15]].
/// 
///     5. Workgroup level layout with order:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
///     ```
///     In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
///     arranged as [[0, 2, 4, 6], [1, 3, 5, 7]]. Each subgroup accesses a 16x16 block per instruction, which
///     is further distributed to 16 work items which is organized as [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]].
/// 
///     6. Workgroup level layout with inst_data:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     This example is similar to the previous ones, but the `inst_data` parameter divides `sg_data` into two instructions,
///     each processing an 8x16 block. These blocks are further distributed across 16 work-items with a distribution unit of 1x1.
///     Unlike the 2x2 distribution unit in example 3, which results in accessing contiguous 2x2 blocks, the 1x1 distribution
///     unit may result in non-contiguous access.
class LayoutAttr;
namespace detail {
struct BlockTensorDescAttrStorage;
} // namespace detail
class BlockTensorDescAttr : public ::mlir::Attribute::AttrBase<BlockTensorDescAttr, ::mlir::Attribute, detail::BlockTensorDescAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.block_tdesc_attr";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static BlockTensorDescAttr get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr array_length, BoolAttr boundary_check);
  static BlockTensorDescAttr get(::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int array_length = 1, bool boundary_check = true);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block_tdesc_attr"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemorySpaceAttr getMemorySpace() const;
  IntegerAttr getArrayLength() const;
  BoolAttr getBoundaryCheck() const;
};
namespace detail {
struct ScatterTensorDescAttrStorage;
} // namespace detail
class ScatterTensorDescAttr : public ::mlir::Attribute::AttrBase<ScatterTensorDescAttr, ::mlir::Attribute, detail::ScatterTensorDescAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.scatter_tdesc_attr";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static ScatterTensorDescAttr get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ScatterTensorDescAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ScatterTensorDescAttr get(::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int chunk_size = 1);
  static ScatterTensorDescAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int chunk_size = 1);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"scatter_tdesc_attr"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemorySpaceAttr getMemorySpace() const;
  IntegerAttr getChunkSize() const;
};
namespace detail {
struct MemorySpaceAttrStorage;
} // namespace detail
class MemorySpaceAttr : public ::mlir::Attribute::AttrBase<MemorySpaceAttr, ::mlir::Attribute, detail::MemorySpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.memory_space";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static MemorySpaceAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::MemorySpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::MemorySpace getValue() const;
};
namespace detail {
struct CachePolicyAttrStorage;
} // namespace detail
class CachePolicyAttr : public ::mlir::Attribute::AttrBase<CachePolicyAttr, ::mlir::Attribute, detail::CachePolicyAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.cache_hint";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static CachePolicyAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::CachePolicy value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cache_hint"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::CachePolicy getValue() const;
};
namespace detail {
struct FenceScopeAttrStorage;
} // namespace detail
class FenceScopeAttr : public ::mlir::Attribute::AttrBase<FenceScopeAttr, ::mlir::Attribute, detail::FenceScopeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.fence_scope";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static FenceScopeAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::FenceScope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fence_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::FenceScope getValue() const;
};
namespace detail {
struct LayoutAttrStorage;
} // namespace detail
class LayoutAttr : public ::mlir::Attribute::AttrBase<LayoutAttr, ::mlir::Attribute, detail::LayoutAttrStorage> {
public:
  using Base::Base;
  bool isWgLayout() {
    return getSgLayout() != nullptr;
  }

  bool isSgLayout() {
    return !isWgLayout();
  }

  int64_t getRank() {
    if (auto attr = getSgLayout())
      return attr.size();
    if (auto attr = getInstData())
      return attr.size();
    if (auto attr = getLaneLayout())
      return attr.size();
    return 0;
  }

  LayoutAttr dropSgLayoutAndData() {
    // avoid every field of the attribute is nullptr, which may lead to segment fault
    if (!getInstData() && !getLaneLayout())
      return nullptr;
    return LayoutAttr::get(getContext(), nullptr, nullptr, getInstData(),
                           getLaneLayout(), getLaneData(), getOrder());
  }

  LayoutAttr dropInstData() {
    // avoid every field of the attribute is nullptr, which may lead to segment fault
    if (!getSgLayout() && !getLaneLayout())
      return nullptr;
    return LayoutAttr::get(getContext(), getSgLayout(), getSgData(), nullptr,
                           getLaneLayout(), getLaneData(), getOrder());
  }
  static constexpr ::llvm::StringLiteral name = "xegpu.layout";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static LayoutAttr get(::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data, llvm::ArrayRef<int32_t> order);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data, llvm::ArrayRef<int32_t> order);
  static LayoutAttr get(::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DenseI32ArrayAttr getSgLayout() const;
  DenseI32ArrayAttr getSgData() const;
  DenseI32ArrayAttr getInstData() const;
  DenseI32ArrayAttr getLaneLayout() const;
  DenseI32ArrayAttr getLaneData() const;
  DenseI32ArrayAttr getOrder() const;
};
} // namespace xegpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::BlockTensorDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::ScatterTensorDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::CachePolicyAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::FenceScopeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::LayoutAttr)

#endif  // GET_ATTRDEF_CLASSES

