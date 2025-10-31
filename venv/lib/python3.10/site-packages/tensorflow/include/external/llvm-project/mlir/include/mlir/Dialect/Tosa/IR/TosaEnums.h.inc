/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOpBase.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
// supported TOSA extensions
enum class Extension : uint32_t {
  none = 0,
  int16 = 1,
  int4 = 2,
  bf16 = 3,
  fp8e4m3 = 4,
  fp8e5m2 = 5,
  fft = 6,
  variable = 7,
  controlflow = 8,
  doubleround = 9,
  inexactround = 10,
  dynamic = 11,
};

::std::optional<Extension> symbolizeExtension(uint32_t);
::llvm::StringRef stringifyExtension(Extension);
::std::optional<Extension> symbolizeExtension(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForExtension() {
  return 11;
}


inline ::llvm::StringRef stringifyEnum(Extension enumValue) {
  return stringifyExtension(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Extension> symbolizeEnum<Extension>(::llvm::StringRef str) {
  return symbolizeExtension(str);
}
} // namespace tosa
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::Extension, ::mlir::tosa::Extension> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::Extension> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for supported TOSA extensions");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Extension> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Extension>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, int16, int4, bf16, fp8e4m3, fp8e5m2, fft, variable, controlflow, doubleround, inexactround, dynamic] for supported TOSA extensions, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::Extension>, std::optional<::mlir::tosa::Extension>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::Extension>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::Extension>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Extension> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Extension>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, int16, int4, bf16, fp8e4m3, fp8e5m2, fft, variable, controlflow, doubleround, inexactround, dynamic] for supported TOSA extensions, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::Extension value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::Extension> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::Extension getEmptyKey() {
    return static_cast<::mlir::tosa::Extension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::Extension getTombstoneKey() {
    return static_cast<::mlir::tosa::Extension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::Extension &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::Extension &lhs, const ::mlir::tosa::Extension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace tosa {
// supported TOSA profiles
enum class Profile : uint32_t {
  pro_int = 1,
  pro_fp = 2,
  none = 0,
};

::std::optional<Profile> symbolizeProfile(uint32_t);
::llvm::StringRef stringifyProfile(Profile);
::std::optional<Profile> symbolizeProfile(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProfile() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Profile enumValue) {
  return stringifyProfile(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Profile> symbolizeEnum<Profile>(::llvm::StringRef str) {
  return symbolizeProfile(str);
}
} // namespace tosa
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::Profile, ::mlir::tosa::Profile> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::Profile> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for supported TOSA profiles");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Profile> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Profile>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [pro_int, pro_fp, none] for supported TOSA profiles, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::Profile>, std::optional<::mlir::tosa::Profile>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::Profile>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::Profile>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Profile> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Profile>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [pro_int, pro_fp, none] for supported TOSA profiles, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::Profile value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::Profile> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::Profile getEmptyKey() {
    return static_cast<::mlir::tosa::Profile>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::Profile getTombstoneKey() {
    return static_cast<::mlir::tosa::Profile>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::Profile &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::Profile &lhs, const ::mlir::tosa::Profile &rhs) {
    return lhs == rhs;
  }
};
}

