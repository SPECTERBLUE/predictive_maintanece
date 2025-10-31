/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Passes.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
::llvm::StringRef stringifyTosaLevelEnum(TosaLevelEnum val) {
  switch (val) {
    case TosaLevelEnum::None: return "none";
    case TosaLevelEnum::EightK: return "8k";
  }
  return "";
}

::std::optional<TosaLevelEnum> symbolizeTosaLevelEnum(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TosaLevelEnum>>(str)
      .Case("none", TosaLevelEnum::None)
      .Case("8k", TosaLevelEnum::EightK)
      .Default(::std::nullopt);
}
::std::optional<TosaLevelEnum> symbolizeTosaLevelEnum(uint32_t value) {
  switch (value) {
  case 0: return TosaLevelEnum::None;
  case 1: return TosaLevelEnum::EightK;
  default: return ::std::nullopt;
  }
}

bool TosaLevelEnumAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
TosaLevelEnumAttr TosaLevelEnumAttr::get(::mlir::MLIRContext *context, TosaLevelEnum val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<TosaLevelEnumAttr>(baseAttr);
}
TosaLevelEnum TosaLevelEnumAttr::getValue() const {
  return static_cast<TosaLevelEnum>(::mlir::IntegerAttr::getValue().getZExtValue());
}
} // namespace tosa
} // namespace mlir

