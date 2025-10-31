/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace triton {
namespace gpu {
class TritonGPU_AttrTrait;
namespace detail {
struct TritonGPU_AttrTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::TritonGPU_AttrTrait;
    Model() : Concept{} {}

  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::TritonGPU_AttrTrait;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct TritonGPU_AttrTraitTrait;

} // namespace detail
class TritonGPU_AttrTrait : public ::mlir::AttributeInterface<TritonGPU_AttrTrait, detail::TritonGPU_AttrTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<TritonGPU_AttrTrait, detail::TritonGPU_AttrTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::TritonGPU_AttrTraitTrait<ConcreteAttr> {};
};
namespace detail {
  template <typename ConcreteAttr>
  struct TritonGPU_AttrTraitTrait : public ::mlir::AttributeInterface<TritonGPU_AttrTrait, detail::TritonGPU_AttrTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
/// Common trait for all TTGIR layouts.
class LayoutEncodingTrait;
namespace detail {
struct LayoutEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getCTAsPerCGA)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getCTAOrder)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getCTASplitNum)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    Model() : Concept{getCTAsPerCGA, getCTAOrder, getCTASplitNum} {}

    static inline SmallVector<unsigned> getCTAsPerCGA(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getCTAOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getCTASplitNum(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    FallbackModel() : Concept{getCTAsPerCGA, getCTAOrder, getCTASplitNum} {}

    static inline SmallVector<unsigned> getCTAsPerCGA(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getCTAOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getCTASplitNum(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct LayoutEncodingTraitTrait;

} // namespace detail
class LayoutEncodingTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::LayoutEncodingTraitTrait<ConcreteAttr> {};

  /// Get the shape of the CTAs per CGA.
  SmallVector<unsigned> getCTAsPerCGA() const;

  /// Get the order of the CTAs per CGA. The fastest-changing axis first
  SmallVector<unsigned> getCTAOrder() const;

  /// Each CTA processes 1/CTASplitNum of the tensor.
  SmallVector<unsigned> getCTASplitNum() const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct LayoutEncodingTraitTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
/// Common trait describing shared memory.
class SharedEncodingTrait;
namespace detail {
struct SharedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int32_t (*getAlignment)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    Model() : Concept{getAlignment} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    FallbackModel() : Concept{getAlignment} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct SharedEncodingTraitTrait;

} // namespace detail
class SharedEncodingTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::SharedEncodingTraitTrait<ConcreteAttr> {};

  /// Return the default alignment for the layout.
  int32_t getAlignment() const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct SharedEncodingTraitTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
/// The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
/// It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.
/// 
/// For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
/// For example, for a shape/order pair defines a distribution layout
/// shape = [4, 4]
/// order = [0, 1] // The fastest-changing axis first
/// ->
/// layout = [0  4  8  12]
///          [1  5  9  13]
///          [2  6  10 14]
///          [3  7  11 15]
/// 
/// For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
/// 
/// If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
/// We call each individual tile "rep".
class DistributedEncodingTrait;
namespace detail {
struct DistributedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrder)(const Concept *impl, ::mlir::Attribute );
    unsigned (*getTotalElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    SmallVector<unsigned> (*getElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    LinearLayout (*toLinearLayout)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    Model() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    FallbackModel() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    unsigned getTotalElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const;
    SmallVector<unsigned> getElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const;
  };
};
template <typename ConcreteAttr>
struct DistributedEncodingTraitTrait;

} // namespace detail
class DistributedEncodingTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DistributedEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrder() const;

  /// Return total element size per thread.
  unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Return element size per thread in each dimension.
  SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Convert to LinearLayout.
  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DistributedEncodingTraitTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Return total element size per thread.
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
      return toLinearEncoding((*static_cast<const ConcreteAttr *>(this)), shape).getTotalElemsPerThread(shape);
    }

    /// Return element size per thread in each dimension.
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const {
      return toLinearEncoding((*static_cast<const ConcreteAttr *>(this)), shape).getElemsPerThread(shape);
    }
  };
}// namespace detail
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
class MmaEncodingTrait;
namespace detail {
struct MmaEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrderForOperand)(const Concept *impl, ::mlir::Attribute , int);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    Model() : Concept{getRepOrderForOperand} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    FallbackModel() : Concept{getRepOrderForOperand} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct MmaEncodingTraitTrait;

} // namespace detail
class MmaEncodingTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::MmaEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct MmaEncodingTraitTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCTAsPerCGA(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCTAsPerCGA();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCTAOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCTAOrder();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCTASplitNum(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCTASplitNum();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCTAsPerCGA(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCTAsPerCGA(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCTAOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCTAOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCTASplitNum(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCTASplitNum(tablegen_opaque_val);
}
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getAlignment();
}
template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getAlignment(tablegen_opaque_val);
}
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrder();
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getTotalElemsPerThread(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getElemsPerThread(shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).toLinearLayout(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getTotalElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->toLinearLayout(tablegen_opaque_val, shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getTotalElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const {
return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getTotalElemsPerThread(shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const {
return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getElemsPerThread(shape);
}
} // namespace gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrderForOperand(opIdx);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrderForOperand(tablegen_opaque_val, opIdx);
}
} // namespace gpu
} // namespace triton
} // namespace mlir
