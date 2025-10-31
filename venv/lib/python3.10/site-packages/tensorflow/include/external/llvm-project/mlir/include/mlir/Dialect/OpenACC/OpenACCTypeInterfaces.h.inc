/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
/// An interface for pointer-like types that point to an OpenACC var.
class PointerLikeType;
namespace detail {
struct PointerLikeTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    ::mlir::acc::VariableTypeCategory (*getPointeeTypeCategory)(const Concept *impl, ::mlir::Type , ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::Type);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::PointerLikeType;
    Model() : Concept{getElementType, getPointeeTypeCategory} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::PointerLikeType;
    FallbackModel() : Concept{getElementType, getPointeeTypeCategory} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const;
  };
};
template <typename ConcreteType>
struct PointerLikeTypeTrait;

} // namespace detail
class PointerLikeType : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::PointerLikeTypeTrait<ConcreteType> {};

  /// Returns the pointee type or null if the pointer has no pointee type
  ::mlir::Type getElementType() const;

  /// Returns the type category of the pointee. The `var` is provided because
  /// a dialect's type system may be incomplete. For example, consider a
  /// dialect which computes interior pointers - so a float array element
  /// may be represented as `ptr<f32>`. The type system says the pointee
  /// is `f32` but this is not a scalar from the point-of-view of OpenACC.
  /// It is an array element and thus the appropriate type category is
  /// "array" - therefore being able to look up how a variable is computed
  /// is important for a complete type determination.
  /// The `varType` is provided in cases where a dialect's type system
  /// erased the target type.
  ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const;
};
namespace detail {
  template <typename ConcreteType>
  struct PointerLikeTypeTrait : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Returns the type category of the pointee. The `var` is provided because
    /// a dialect's type system may be incomplete. For example, consider a
    /// dialect which computes interior pointers - so a float array element
    /// may be represented as `ptr<f32>`. The type system says the pointee
    /// is `f32` but this is not a scalar from the point-of-view of OpenACC.
    /// It is an array element and thus the appropriate type category is
    /// "array" - therefore being able to look up how a variable is computed
    /// is important for a complete type determination.
    /// The `varType` is provided in cases where a dialect's type system
    /// erased the target type.
    ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const {
      return ::mlir::acc::VariableTypeCategory::uncategorized;
    }
  };
}// namespace detail
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
/// An interface to capture type-based semantics for mapping in a manner that
///     makes it convertible to size-based semantics.
class MappableType;
namespace detail {
struct MappableTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> (*getVarPtr)(const Concept *impl, ::mlir::Type , ::mlir::Value);
    ::std::optional<::llvm::TypeSize> (*getSizeInBytes)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::ValueRange, const ::mlir::DataLayout &);
    ::std::optional<::int64_t> (*getOffsetInBytes)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::ValueRange, const ::mlir::DataLayout &);
    ::llvm::SmallVector<::mlir::Value> (*generateAccBounds)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::OpBuilder &);
    ::mlir::acc::VariableTypeCategory (*getTypeCategory)(const Concept *impl, ::mlir::Type , ::mlir::Value);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::MappableType;
    Model() : Concept{getVarPtr, getSizeInBytes, getOffsetInBytes, generateAccBounds, getTypeCategory} {}

    static inline ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::std::optional<::llvm::TypeSize> getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::std::optional<::int64_t> getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::llvm::SmallVector<::mlir::Value> generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder);
    static inline ::mlir::acc::VariableTypeCategory getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::MappableType;
    FallbackModel() : Concept{getVarPtr, getSizeInBytes, getOffsetInBytes, generateAccBounds, getTypeCategory} {}

    static inline ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::std::optional<::llvm::TypeSize> getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::std::optional<::int64_t> getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::llvm::SmallVector<::mlir::Value> generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder);
    static inline ::mlir::acc::VariableTypeCategory getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
    ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const;
    ::std::optional<::int64_t> getOffsetInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const;
    ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder &builder) const;
    ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
  };
};
template <typename ConcreteType>
struct MappableTypeTrait;

} // namespace detail
class MappableType : public ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::MappableTypeTrait<ConcreteType> {};

  /// Returns the pointer to the `var` if recoverable (such as in cases
  /// where the current operation is a load from a memory slot).
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Value var) const;

  /// Returns the size in bytes when computable. If this is an array-like
  /// type, avoiding passing `accBounds` ensures a computation of the size
  /// of whole type.
  ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const;

  /// Returns the offset in bytes when computable.
  ::std::optional<::int64_t> getOffsetInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const;

  /// Returns explicit `acc.bounds` operations that envelop the whole
  /// data structure. These operations are inserted using the provided builder
  /// at the location set before calling this API.
  ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Value var, ::mlir::OpBuilder & builder) const;

  /// Returns the OpenACC type category.
  ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Value var) const;
};
namespace detail {
  template <typename ConcreteType>
  struct MappableTypeTrait : public ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Returns the pointer to the `var` if recoverable (such as in cases
    /// where the current operation is a load from a memory slot).
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Value var) const {
      if (auto ptr = mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
              var))
          return ptr;
        return {};
    }

    /// Returns the size in bytes when computable. If this is an array-like
    /// type, avoiding passing `accBounds` ensures a computation of the size
    /// of whole type.
    ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};
        return {dataLayout.getTypeSize((*static_cast<const ConcreteType *>(this)))};
    }

    /// Returns the offset in bytes when computable.
    ::std::optional<::int64_t> getOffsetInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};

        // If the type size is computable, it means it is trivial. Assume
        // offset of 0.
        if (::mlir::cast<::mlir::acc::MappableType>((*static_cast<const ConcreteType *>(this))).getSizeInBytes(
              var, accBounds, dataLayout).has_value()) {
          return {0};
        }

        return {};
    }

    /// Returns explicit `acc.bounds` operations that envelop the whole
    /// data structure. These operations are inserted using the provided builder
    /// at the location set before calling this API.
    ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Value var, ::mlir::OpBuilder & builder) const {
      return {};
    }

    /// Returns the OpenACC type category.
    ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Value var) const {
      return ::mlir::acc::VariableTypeCategory::uncategorized;
    }
  };
}// namespace detail
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getPointeeTypeCategory(varPtr, varType);
}
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) {
  return static_cast<const ConcreteType *>(impl)->getPointeeTypeCategory(tablegen_opaque_val, varPtr, varType);
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getPointeeTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const {
return ::mlir::acc::VariableTypeCategory::uncategorized;
}
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
template<typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getVarPtr(var);
}
template<typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSizeInBytes(var, accBounds, dataLayout);
}
template<typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getOffsetInBytes(var, accBounds, dataLayout);
}
template<typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generateAccBounds(var, builder);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getTypeCategory(var);
}
template<typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->getVarPtr(tablegen_opaque_val, var);
}
template<typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteType *>(impl)->getSizeInBytes(tablegen_opaque_val, var, accBounds, dataLayout);
}
template<typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteType *>(impl)->getOffsetInBytes(tablegen_opaque_val, var, accBounds, dataLayout);
}
template<typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteType *>(impl)->generateAccBounds(tablegen_opaque_val, var, builder);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->getTypeCategory(tablegen_opaque_val, var);
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getVarPtr(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
if (auto ptr = mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
              var))
          return ptr;
        return {};
}
template<typename ConcreteModel, typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getSizeInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const {
// Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};
        return {dataLayout.getTypeSize((::llvm::cast<ConcreteType>(tablegen_opaque_val)))};
}
template<typename ConcreteModel, typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getOffsetInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const {
// Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};

        // If the type size is computable, it means it is trivial. Assume
        // offset of 0.
        if (::mlir::cast<::mlir::acc::MappableType>((::llvm::cast<ConcreteType>(tablegen_opaque_val))).getSizeInBytes(
              var, accBounds, dataLayout).has_value()) {
          return {0};
        }

        return {};
}
template<typename ConcreteModel, typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generateAccBounds(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder &builder) const {
return {};
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
return ::mlir::acc::VariableTypeCategory::uncategorized;
}
} // namespace acc
} // namespace mlir
