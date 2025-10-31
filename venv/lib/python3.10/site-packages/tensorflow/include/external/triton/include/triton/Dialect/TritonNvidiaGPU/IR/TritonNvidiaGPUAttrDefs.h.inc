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
namespace triton {
namespace nvidia_gpu {
/// Attribute to indicate that the memory descriptor points to tensor memory.
///     The memory is laid out in blocks of size blockM x blockN. Each block is distributed
///     across TMEM 128 rows.
/// 
///     Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
///     convention that need to be followed operations reading/writing to TMEM.
/// 
///     a tensor <128x128xf32> with blockM = 64 and blockN = 64 will be distributed as follows:
/// 
///         \ col    0        1           31       32            64             96           127
///     rows: 0  ( 0,  0) ( 0,  1) ... ( 0, 31)  (64,  0) ...   (0, 64) ...  (64, 64) ...  (64, 96)
///           1
///          ...
///           15 (15,  0) (15,  1) ... (15, 31)  (79,  0) ...   (15, 64) ... (79, 64) ...  (79, 96)
///           16 ( 0, 32) ( 0, 33) ... ( 0, 63)  (64, 32) ...   ( 0, 96) ... (64, 96) ...  (64, 127)
///          ...
///           31 (15, 32) (15, 33) ... (15, 63)  (79, 32) ...   (15, 96) ... (79, 96) ...  (79, 127)
///           32 (16,  0) (16,  1) ... (16, 31)  (80,  0) ...   (16, 64) ... (80, 64) ...  (80, 96)
///          ...
///          127 (63, 32) (63, 33) ... (63, 63)  (127, 32) ...  (63, 96) ... (127, 96)...  (127, 127)
class TensorMemorySpaceAttr;
/// An encoding to represent the different way the tensor memory is laid out.
///     `unpacked` attributes indicates whether types smaller than 32bits are unpacked (take full 32bits)
///     or are packed (N elements are stored within one 32bits row).
class TensorMemoryEncodingAttr;
/// An encoding to represent the layout of tensor memory scales.
///     As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
///     as a multiple copies of "chunk", each of which having the size 32x4x4B. Moreover, such chunks are duplicated
///     over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
///     layout.
class TensorMemoryScalesEncodingAttr;
class TensorMemorySpaceAttr : public ::mlir::Attribute::AttrBase<TensorMemorySpaceAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.tensor_memory";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory"};
  }

};
namespace detail {
struct TensorMemoryEncodingAttrStorage;
} // namespace detail
class TensorMemoryEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryEncodingAttr, ::mlir::Attribute, detail::TensorMemoryEncodingAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static TensorMemoryEncodingAttr get(::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, bool unpacked, unsigned CTASplitM, unsigned CTASplitN);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getBlockM() const;
  unsigned getBlockN() const;
  bool getUnpacked() const;
  unsigned getCTASplitM() const;
  unsigned getCTASplitN() const;
};
namespace detail {
struct TensorMemoryScalesEncodingAttrStorage;
} // namespace detail
class TensorMemoryScalesEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryScalesEncodingAttr, ::mlir::Attribute, detail::TensorMemoryScalesEncodingAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_scales_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static TensorMemoryScalesEncodingAttr get(::mlir::MLIRContext *context, unsigned CTASplitM, unsigned CTASplitN);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_scales_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getCTASplitM() const;
  unsigned getCTASplitN() const;
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr)

#endif  // GET_ATTRDEF_CLASSES

