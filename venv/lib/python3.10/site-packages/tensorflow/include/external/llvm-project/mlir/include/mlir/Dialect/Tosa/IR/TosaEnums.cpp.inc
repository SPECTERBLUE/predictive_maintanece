/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOpBase.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
::llvm::StringRef stringifyExtension(Extension val) {
  switch (val) {
    case Extension::none: return "none";
    case Extension::int16: return "int16";
    case Extension::int4: return "int4";
    case Extension::bf16: return "bf16";
    case Extension::fp8e4m3: return "fp8e4m3";
    case Extension::fp8e5m2: return "fp8e5m2";
    case Extension::fft: return "fft";
    case Extension::variable: return "variable";
    case Extension::controlflow: return "controlflow";
    case Extension::doubleround: return "doubleround";
    case Extension::inexactround: return "inexactround";
    case Extension::dynamic: return "dynamic";
  }
  return "";
}

::std::optional<Extension> symbolizeExtension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Extension>>(str)
      .Case("none", Extension::none)
      .Case("int16", Extension::int16)
      .Case("int4", Extension::int4)
      .Case("bf16", Extension::bf16)
      .Case("fp8e4m3", Extension::fp8e4m3)
      .Case("fp8e5m2", Extension::fp8e5m2)
      .Case("fft", Extension::fft)
      .Case("variable", Extension::variable)
      .Case("controlflow", Extension::controlflow)
      .Case("doubleround", Extension::doubleround)
      .Case("inexactround", Extension::inexactround)
      .Case("dynamic", Extension::dynamic)
      .Default(::std::nullopt);
}
::std::optional<Extension> symbolizeExtension(uint32_t value) {
  switch (value) {
  case 0: return Extension::none;
  case 1: return Extension::int16;
  case 2: return Extension::int4;
  case 3: return Extension::bf16;
  case 4: return Extension::fp8e4m3;
  case 5: return Extension::fp8e5m2;
  case 6: return Extension::fft;
  case 7: return Extension::variable;
  case 8: return Extension::controlflow;
  case 9: return Extension::doubleround;
  case 10: return Extension::inexactround;
  case 11: return Extension::dynamic;
  default: return ::std::nullopt;
  }
}

} // namespace tosa
} // namespace mlir

namespace mlir {
namespace tosa {
::llvm::StringRef stringifyProfile(Profile val) {
  switch (val) {
    case Profile::pro_int: return "pro_int";
    case Profile::pro_fp: return "pro_fp";
    case Profile::none: return "none";
  }
  return "";
}

::std::optional<Profile> symbolizeProfile(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Profile>>(str)
      .Case("pro_int", Profile::pro_int)
      .Case("pro_fp", Profile::pro_fp)
      .Case("none", Profile::none)
      .Default(::std::nullopt);
}
::std::optional<Profile> symbolizeProfile(uint32_t value) {
  switch (value) {
  case 1: return Profile::pro_int;
  case 2: return Profile::pro_fp;
  case 0: return Profile::none;
  default: return ::std::nullopt;
  }
}

} // namespace tosa
} // namespace mlir

