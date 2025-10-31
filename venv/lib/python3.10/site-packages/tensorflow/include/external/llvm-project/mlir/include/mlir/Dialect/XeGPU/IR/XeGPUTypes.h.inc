/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace xegpu {
/// TensorDesc describing regions of interested data.
/// TensorDesc is a type designed to describe regions of interest in data, as well as some features
///     unique to Intel hardware. Unlike the built-in tensor type in MLIR, it essentially contains only
///     metadata and does not hold the data itself. It is primarily designed to support 2D block load/store
///     and DPAS (matrix multiplication instruction) on Intel GPUs. It encodes the following information:
/// 
///     * shape:  the sizes/shape of the interested data block, e.g., 8x16 means 8 rows
///               and each row contains 16 contiguous data element. The rows could be
///               either contiguous or not, depends on the encoding attribute. If the
///               encoding is a BlockTensorDescAttr, rows are contiguous. If the encoding
///               is a ScatterTensorDescAttr, rows are not necessary to be contiguous. If
///               encoding is not set, it is considered as a default BlockTensorDescAttr.
/// 
///     * element_type: the data type of the data element, e.g., f16, f32.
/// 
///     Similar to the built-in tensor, it also provides optional attributes for encoding
///     additional information via either BlockTensorDescAttr or ScatterTensorDescAttr, or
///     supporting Workgroup, Subgroup, and workitem (or SIMT) level programmings via the
///     Layout attribute. Please check their definition for details.
/// 
///     Syntax:
/// 
///     ```
///     TensorDesc-type ::= `tensor_desc` `<` dim-list element-type (attr-list)? `>`
///     element-type ::= float-type | integer-type | index-type
///     dim-list := (static-dim-list `x`)?
///     static-dim-list ::= decimal-literal `x` decimal-literal
///     attr-list = (, encoding-attr)? (, layout-attr)?
///     enconding-attr = (, memory_space = value)? (, arr_len = value)? (, boundary_check = value)? (, scattered = value)?
///     layout-attr = (, layout `<`sg_layout = value, sg_data = value, inst_data = value, lane_layout = value, lane_data = value, order = value`>`)?
///     ```
/// 
///     Examples:
/// 
///     ```mlir
///     // A block TensorDesc with 8x16 i32 elements
///     xegpu.tensor_desc<8x16xi32>
/// 
///     // A block TensorDesc with 8x16 f32 elements
///     xegpu.tensor_desc<8x16xf32>
/// 
///     // A TensorDesc with 8x16 f32 elements for a memory region in shared memory space.
///     xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_space = slm>>
/// 
///     // A 1D TensorDesc with a layout for subgroup level programming, each lane access two continuous elements
///     xegpu.tensor_desc<32xf32, #xegpu.layout<lane_layout = [16], lane_data = [2]>>
/// 
///     // A 1D TensorDesc with a layout for subgroup level programming, each lane access two elements with stride = 16
///     xegpu.tensor_desc<32xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
/// 
///     // A TensorDesc with a layout for subgroup level programming
///     xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
/// 
///     // A TensorDesc with a layout for workgroup level programming
///     xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
/// 
///     // A TensorDesc with a layout for workgroup level programming without lane_layout and lane_data
///     xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16]>>
/// 
///     ```
class TensorDescType;
/// !xegpu.nbarrier a custom XeGPU type representing a barrier.
class NbarrierType;
namespace detail {
struct TensorDescTypeStorage;
} // namespace detail
class TensorDescType : public ::mlir::Type::TypeBase<TensorDescType, ::mlir::TensorType, detail::TensorDescTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  using TensorType::clone;
  using mlir::ShapedType::Trait<TensorDescType>::getElementTypeBitWidth;
  using mlir::ShapedType::Trait<TensorDescType>::getRank;
  using mlir::ShapedType::Trait<TensorDescType>::getNumElements;
  using mlir::ShapedType::Trait<TensorDescType>::isDynamicDim;
  using mlir::ShapedType::Trait<TensorDescType>::hasStaticShape;
  using mlir::ShapedType::Trait<TensorDescType>::getNumDynamicDims;
  using mlir::ShapedType::Trait<TensorDescType>::getDimSize;
  using mlir::ShapedType::Trait<TensorDescType>::getDynamicDimIndex;

  TensorDescType clone(::mlir::Type elementType) {
    return llvm::cast<TensorDescType>(cloneWith(getShape(), elementType));
  }

  BlockTensorDescAttr getEncodingAsBlockTensorDescAttr() const {
    return llvm::dyn_cast_if_present<BlockTensorDescAttr>(getEncoding());
  }

  ScatterTensorDescAttr getEncodingAsScatterTensorDescAttr() const {
    return llvm::dyn_cast_if_present<ScatterTensorDescAttr>(getEncoding());
  }

  LayoutAttr getLayoutAttr() const {
    return llvm::dyn_cast_if_present<LayoutAttr>(getLayout());
  }

  xegpu::MemorySpace getMemorySpace() const {
    auto block_attr = getEncodingAsBlockTensorDescAttr();
    if (block_attr && block_attr.getMemorySpace())
      return block_attr.getMemorySpace().getValue();

    auto scatter_attr = getEncodingAsScatterTensorDescAttr();
    if (scatter_attr && scatter_attr.getMemorySpace())
      return scatter_attr.getMemorySpace().getValue();

    // return default value
    return MemorySpace::Global;
  }

  int getArrayLength() {
    auto attr = getEncoding();
    auto block_attr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(attr);
    assert((!attr || block_attr) && "invalid on non BlockTensorDescAttr.");
    if (block_attr && block_attr.getArrayLength())
      return block_attr.getArrayLength().getInt();
    // return default value
    return 1;
  }

  bool getBoundaryCheck() {
    auto attr = getEncoding();
    auto block_attr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(attr);
    assert((!attr || block_attr) && "invalid on non BlockTensorDescAttr.");
    if (block_attr && block_attr.getBoundaryCheck())
      return block_attr.getBoundaryCheck().getValue();
    // return default value
    return true;
  }

  bool isScattered() {
    return bool(getEncodingAsScatterTensorDescAttr());
  }

  int getChunkSize() {
    auto attr = getEncoding();
    auto scatter_attr = mlir::dyn_cast_if_present<ScatterTensorDescAttr>(attr);
    assert((!attr || scatter_attr) && "invalid on non ScatterTensorDescAttr.");
    if (scatter_attr)
      return scatter_attr.getChunkSize().getInt();
    return 1;
  }

  /// Helper to drop all layout information from the TensorDesc type.
  TensorDescType dropLayouts() {
    if (!getLayoutAttr())
      return *this;

    return get(getContext(), getShape(), getElementType(), getEncoding(),
      xegpu::LayoutAttr());
  }
  static constexpr ::llvm::StringLiteral name = "xegpu.tensor_desc";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static TensorDescType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout);
  static TensorDescType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout);
  static TensorDescType get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType, int array_length = 1, bool boundary_check = true, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, mlir::Attribute layout = mlir::Attribute());
  static TensorDescType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, llvm::ArrayRef<int64_t> shape, mlir::Type elementType, int array_length = 1, bool boundary_check = true, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, mlir::Attribute layout = mlir::Attribute());
  static TensorDescType get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType, int chunk_size = 1, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, mlir::Attribute layout = mlir::Attribute());
  static TensorDescType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, llvm::ArrayRef<int64_t> shape, mlir::Type elementType, int chunk_size = 1, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, mlir::Attribute layout = mlir::Attribute());
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_desc"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
  mlir::Attribute getEncoding() const;
  mlir::Attribute getLayout() const;
};
class NbarrierType : public ::mlir::Type::TypeBase<NbarrierType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static NbarrierType get(mlir::MLIRContext *context) {
    return Base::get(context);
  };
  static constexpr ::llvm::StringLiteral name = "xegpu.nbarrier";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nbarrier"};
  }

};
} // namespace xegpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::TensorDescType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::NbarrierType)

#endif  // GET_TYPEDEF_CLASSES

