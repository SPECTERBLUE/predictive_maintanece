/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonNvidiaGPUOps.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::nvidia_gpu::ArriveBarrierOp,
::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp,
::mlir::triton::nvidia_gpu::AsyncTMAGatherOp,
::mlir::triton::nvidia_gpu::AsyncTMAReduceOp,
::mlir::triton::nvidia_gpu::AsyncTMAScatterOp,
::mlir::triton::nvidia_gpu::BarrierExpectOp,
::mlir::triton::nvidia_gpu::ClusterArriveOp,
::mlir::triton::nvidia_gpu::ClusterWaitOp,
::mlir::triton::nvidia_gpu::FenceAsyncSharedOp,
::mlir::triton::nvidia_gpu::InitBarrierOp,
::mlir::triton::nvidia_gpu::InvalBarrierOp,
::mlir::triton::nvidia_gpu::TCGen5MMAOp,
::mlir::triton::nvidia_gpu::TCGen5MMAScaledOp,
::mlir::triton::nvidia_gpu::TMAStoreWaitOp,
::mlir::triton::nvidia_gpu::TMEMAllocOp,
::mlir::triton::nvidia_gpu::TMEMCopyOp,
::mlir::triton::nvidia_gpu::TMEMLoadOp,
::mlir::triton::nvidia_gpu::TMEMStoreOp,
::mlir::triton::nvidia_gpu::TMEMSubSliceOp,
::mlir::triton::nvidia_gpu::TensorDescToTMAPtrOp,
::mlir::triton::nvidia_gpu::WaitBarrierOp,
::mlir::triton::nvidia_gpu::WarpGroupDotOp,
::mlir::triton::nvidia_gpu::WarpGroupDotWaitOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace nvidia_gpu {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::MemDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memory descriptor type (`::mlir::triton::gpu::MemDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::PointerType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Pointer type (`::mlir::triton::PointerType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::AsyncTokenType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be async token type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::MemDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of memory descriptor type (`::mlir::triton::gpu::MemDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((::llvm::isa<::mlir::Float8E4M3FNType>(elementType))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2Type>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(elementType))) || ((elementType.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(elementType))) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((::mlir::isa<::mlir::triton::PointerType>(elementType))) && ([](::mlir::Type pointeeType) { return (true); }(::mlir::cast<::mlir::triton::PointerType>(elementType).getPointeeType()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of floating-point or integer or ptr values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::TensorDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Tensor descriptor type (`::mlir::triton::TensorDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::isa<::mlir::triton::PointerType>(type))) && ([](::mlir::Type pointeeType) { return (true); }(::mlir::cast<::mlir::triton::PointerType>(type).getPointeeType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ptr, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::TensorOrMemDesc>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be TensorOrMemDesc instance, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((::llvm::isa<::mlir::Float8E4M3FNType>(elementType))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2Type>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(elementType))) || ((elementType.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(elementType))) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of floating-point or integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::TensorOrMemDesc>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of TensorOrMemDesc instance, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::CacheModifierAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6, 7";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::EvictionPolicyAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::DescriptorReduceKindAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6, 7, 8";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(1)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 1-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::UnitAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: unit attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::ScaleDotElemTypeAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps9(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::InputPrecisionAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps9(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ArriveBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ArriveBarrierOpGenericAdaptorBase::ArriveBarrierOpGenericAdaptorBase(ArriveBarrierOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ArriveBarrierOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

uint32_t ArriveBarrierOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ArriveBarrierOpAdaptor::ArriveBarrierOpAdaptor(ArriveBarrierOp op) : ArriveBarrierOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ArriveBarrierOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'ttng.arrive_barrier' op ""requires attribute 'count'");

  if (tblgen_count && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.arrive_barrier' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ArriveBarrierOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange ArriveBarrierOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult ArriveBarrierOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.count;
       auto attr = dict.get("count");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `count` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ArriveBarrierOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ArriveBarrierOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ArriveBarrierOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void ArriveBarrierOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "count") {
       prop.count = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void ArriveBarrierOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.count) attrs.append("count", prop.count);
}

::llvm::LogicalResult ArriveBarrierOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "count", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ArriveBarrierOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.count)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ArriveBarrierOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.count);
}

uint32_t ArriveBarrierOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

void ArriveBarrierOp::setCount(uint32_t attrValue) {
  getProperties().count = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void ArriveBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value alloc, uint32_t count) {
      return build(odsBuilder, odsState, alloc, count, /*pred=*/Value());
    
}

void ArriveBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr count, /*optional*/::mlir::Value pred) {
  odsState.addOperands(alloc);
  if (pred)
    odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().count = count;
}

void ArriveBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr count, /*optional*/::mlir::Value pred) {
  odsState.addOperands(alloc);
  if (pred)
    odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().count = count;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArriveBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t count, /*optional*/::mlir::Value pred) {
  odsState.addOperands(alloc);
  if (pred)
    odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
}

void ArriveBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t count, /*optional*/::mlir::Value pred) {
  odsState.addOperands(alloc);
  if (pred)
    odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArriveBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ArriveBarrierOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ArriveBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ArriveBarrierOp::verifyInvariantsImpl() {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_count, "count")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ArriveBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ArriveBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> predOperands;
  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<ArriveBarrierOp::Properties>().count = countAttr;
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    predOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      predOperands.push_back(operand);
    }
  }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(allocRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType0, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ArriveBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  if (getPred()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getPred())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("count");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getAlloc().getType();
}

void ArriveBarrierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ArriveBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(AsyncTMACopyGlobalToLocalOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::triton::CacheModifierAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getCacheAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(getProperties().cache);
  return attr;
}

::mlir::triton::CacheModifier AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getCache() {
  auto attr = getCacheAttr();
  return attr.getValue();
}

::mlir::triton::EvictionPolicyAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getEvictAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(getProperties().evict);
  return attr;
}

::mlir::triton::EvictionPolicy AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getEvict() {
  auto attr = getEvictAttr();
  return attr.getValue();
}

::mlir::BoolAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getIsVolatileAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().isVolatile);
  return attr;
}

bool AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue();
}

} // namespace detail
AsyncTMACopyGlobalToLocalOpAdaptor::AsyncTMACopyGlobalToLocalOpAdaptor(AsyncTMACopyGlobalToLocalOp op) : AsyncTMACopyGlobalToLocalOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_cache = getProperties().cache; (void)tblgen_cache;
  auto tblgen_evict = getProperties().evict; (void)tblgen_evict;
  auto tblgen_isVolatile = getProperties().isVolatile; (void)tblgen_isVolatile;

  if (tblgen_cache && !((::llvm::isa<::mlir::triton::CacheModifierAttr>(tblgen_cache))))
    return emitError(loc, "'ttng.async_tma_copy_global_to_local' op ""attribute 'cache' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6, 7");

  if (tblgen_evict && !((::llvm::isa<::mlir::triton::EvictionPolicyAttr>(tblgen_evict))))
    return emitError(loc, "'ttng.async_tma_copy_global_to_local' op ""attribute 'evict' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3");

  if (tblgen_isVolatile && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isVolatile))))
    return emitError(loc, "'ttng.async_tma_copy_global_to_local' op ""attribute 'isVolatile' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsyncTMACopyGlobalToLocalOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AsyncTMACopyGlobalToLocalOp::getCoordMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.cache;
       auto attr = dict.get("cache");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cache` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.evict;
       auto attr = dict.get("evict");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `evict` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.isVolatile;
       auto attr = dict.get("isVolatile");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isVolatile` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AsyncTMACopyGlobalToLocalOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.cache;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cache",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.evict;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("evict",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.isVolatile;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isVolatile",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMACopyGlobalToLocalOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.cache.getAsOpaquePointer()), 
    llvm::hash_value(prop.evict.getAsOpaquePointer()), 
    llvm::hash_value(prop.isVolatile.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AsyncTMACopyGlobalToLocalOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "cache")
      return prop.cache;

    if (name == "evict")
      return prop.evict;

    if (name == "isVolatile")
      return prop.isVolatile;
  return std::nullopt;
}

void AsyncTMACopyGlobalToLocalOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "cache") {
       prop.cache = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cache)>>(value);
       return;
    }

    if (name == "evict") {
       prop.evict = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.evict)>>(value);
       return;
    }

    if (name == "isVolatile") {
       prop.isVolatile = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isVolatile)>>(value);
       return;
    }
}

void AsyncTMACopyGlobalToLocalOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.cache) attrs.append("cache", prop.cache);

    if (prop.evict) attrs.append("evict", prop.evict);

    if (prop.isVolatile) attrs.append("isVolatile", prop.isVolatile);
}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCacheAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(attr, "cache", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getEvictAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "evict", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIsVolatileAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, "isVolatile", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.cache)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.evict)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.isVolatile)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyGlobalToLocalOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.cache);

  writer.writeOptionalAttribute(prop.evict);

  writer.writeOptionalAttribute(prop.isVolatile);
}

::mlir::triton::CacheModifier AsyncTMACopyGlobalToLocalOp::getCache() {
  auto attr = getCacheAttr();
  return attr.getValue();
}

::mlir::triton::EvictionPolicy AsyncTMACopyGlobalToLocalOp::getEvict() {
  auto attr = getEvictAttr();
  return attr.getValue();
}

bool AsyncTMACopyGlobalToLocalOp::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue();
}

void AsyncTMACopyGlobalToLocalOp::setCache(::mlir::triton::CacheModifier attrValue) {
  getProperties().cache = ::mlir::triton::CacheModifierAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void AsyncTMACopyGlobalToLocalOp::setEvict(::mlir::triton::EvictionPolicy attrValue) {
  getProperties().evict = ::mlir::triton::EvictionPolicyAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void AsyncTMACopyGlobalToLocalOp::setIsVolatile(bool attrValue) {
  getProperties().isVolatile = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict, ::mlir::BoolAttr isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  if (cache) {
    odsState.getOrAddProperties<Properties>().cache = cache;
  }
  if (evict) {
    odsState.getOrAddProperties<Properties>().evict = evict;
  }
  if (isVolatile) {
    odsState.getOrAddProperties<Properties>().isVolatile = isVolatile;
  }
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict, ::mlir::BoolAttr isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  if (cache) {
    odsState.getOrAddProperties<Properties>().cache = cache;
  }
  if (evict) {
    odsState.getOrAddProperties<Properties>().evict = evict;
  }
  if (isVolatile) {
    odsState.getOrAddProperties<Properties>().isVolatile = isVolatile;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache, ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), cache);
  odsState.getOrAddProperties<Properties>().evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), evict);
  odsState.getOrAddProperties<Properties>().isVolatile = odsBuilder.getBoolAttr(isVolatile);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache, ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), cache);
  odsState.getOrAddProperties<Properties>().evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), evict);
  odsState.getOrAddProperties<Properties>().isVolatile = odsBuilder.getBoolAttr(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyGlobalToLocalOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.cache)
    properties.cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), triton::CacheModifier::NONE);
  if (!properties.evict)
    properties.evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), triton::EvictionPolicy::NORMAL);
  if (!properties.isVolatile)
    properties.isVolatile = odsBuilder.getBoolAttr(false);
}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInvariantsImpl() {
  auto tblgen_cache = getProperties().cache; (void)tblgen_cache;
  auto tblgen_evict = getProperties().evict; (void)tblgen_evict;
  auto tblgen_isVolatile = getProperties().isVolatile; (void)tblgen_isVolatile;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(*this, tblgen_cache, "cache")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_evict, "evict")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(*this, tblgen_isVolatile, "isVolatile")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AsyncTMACopyGlobalToLocalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> coordOperands;
  ::llvm::SMLoc coordOperandsLoc;
  (void)coordOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand resultRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> resultOperands(&resultRawOperand, 1);  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand barrierRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> barrierOperands(&barrierRawOperand, 1);  ::llvm::SMLoc barrierOperandsLoc;
  (void)barrierOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::triton::CacheModifierAttr cacheAttr;
  ::mlir::triton::EvictionPolicyAttr evictAttr;
  ::mlir::Type desc_ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> desc_ptrTypes(&desc_ptrRawType, 1);
  ::mlir::Type barrierRawType{};
  ::llvm::ArrayRef<::mlir::Type> barrierTypes(&barrierRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  coordOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(coordOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  resultOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(resultRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  barrierOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(barrierRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  bool cacheModifierClause = false;
  bool evictionPolicyClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("cacheModifier"))) {

  if (cacheModifierClause) {
    return parser.emitError(parser.getNameLoc())
          << "`cacheModifier` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  cacheModifierClause = true;
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"none","ca","cg","wb","cs","wt","cv"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "cache", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'cache' [none, ca, cg, wb, cs, wt, cv]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeCacheModifier(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "cache attribute specification: \"" << attrStr << '"';;

      cacheAttr = ::mlir::triton::CacheModifierAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>().cache = cacheAttr;
    }
  }
    } else if (succeeded(parser.parseOptionalKeyword("evictionPolicy"))) {

  if (evictionPolicyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`evictionPolicy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  evictionPolicyClause = true;
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"evict_normal","evict_first","evict_last"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "evict", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'evict' [evict_normal, evict_first, evict_last]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeEvictionPolicy(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "evict attribute specification: \"" << attrStr << '"';;

      evictAttr = ::mlir::triton::EvictionPolicyAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>().evict = evictAttr;
    }
  }
    } else  {
    break;
  }
}
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(desc_ptrRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(barrierRawType))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(desc_ptrOperands, desc_ptrTypes, desc_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordOperands, odsBuildableType0, coordOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barrierOperands, barrierTypes, barrierOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyGlobalToLocalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getCoord();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getResult();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBarrier();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  if (false || (getCacheAttr() != ::mlir::triton::CacheModifierAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), triton::CacheModifier::NONE))) {
  _odsPrinter << ' ' << "cacheModifier";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getCache();
    auto caseValueStr = stringifyCacheModifier(caseValue);
    _odsPrinter << caseValueStr;
  }
  }
  if (false || (getEvictAttr() != ::mlir::triton::EvictionPolicyAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), triton::EvictionPolicy::NORMAL))) {
  _odsPrinter << ' ' << "evictionPolicy";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getEvict();
    auto caseValueStr = stringifyEvictionPolicy(caseValue);
    _odsPrinter << caseValueStr;
  }
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("cache");
  elidedAttrs.push_back("evict");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCacheAttr();
     if(attr && (attr == ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), triton::CacheModifier::NONE)))
       elidedAttrs.push_back("cache");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getEvictAttr();
     if(attr && (attr == ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), triton::EvictionPolicy::NORMAL)))
       elidedAttrs.push_back("evict");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsVolatileAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("isVolatile");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getDescPtr().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getBarrier().getType();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void AsyncTMACopyGlobalToLocalOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(2);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(3);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMACopyLocalToGlobalOpGenericAdaptorBase::AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(AsyncTMACopyLocalToGlobalOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AsyncTMACopyLocalToGlobalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
AsyncTMACopyLocalToGlobalOpAdaptor::AsyncTMACopyLocalToGlobalOpAdaptor(AsyncTMACopyLocalToGlobalOp op) : AsyncTMACopyLocalToGlobalOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AsyncTMACopyLocalToGlobalOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsyncTMACopyLocalToGlobalOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AsyncTMACopyLocalToGlobalOp::getCoordMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AsyncTMACopyLocalToGlobalOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute AsyncTMACopyLocalToGlobalOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMACopyLocalToGlobalOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> AsyncTMACopyLocalToGlobalOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void AsyncTMACopyLocalToGlobalOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void AsyncTMACopyLocalToGlobalOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult AsyncTMACopyLocalToGlobalOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AsyncTMACopyLocalToGlobalOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMACopyLocalToGlobalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AsyncTMACopyLocalToGlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> coordOperands;
  ::llvm::SMLoc coordOperandsLoc;
  (void)coordOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type desc_ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> desc_ptrTypes(&desc_ptrRawType, 1);
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  coordOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(coordOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(desc_ptrRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(srcRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(desc_ptrOperands, desc_ptrTypes, desc_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordOperands, odsBuildableType0, coordOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyLocalToGlobalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getCoord();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getDescPtr().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getSrc().getType();
}

void AsyncTMACopyLocalToGlobalOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(2);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMAGatherOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMAGatherOpGenericAdaptorBase::AsyncTMAGatherOpGenericAdaptorBase(AsyncTMAGatherOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

} // namespace detail
AsyncTMAGatherOpAdaptor::AsyncTMAGatherOpAdaptor(AsyncTMAGatherOp op) : AsyncTMAGatherOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AsyncTMAGatherOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAGatherOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute AsyncTMAGatherOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMAGatherOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> AsyncTMAGatherOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void AsyncTMAGatherOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void AsyncTMAGatherOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult AsyncTMAGatherOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void AsyncTMAGatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(x_offsets);
  odsState.addOperands(y_offset);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
}

void AsyncTMAGatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(x_offsets);
  odsState.addOperands(y_offset);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMAGatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsyncTMAGatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AsyncTMAGatherOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAGatherOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AsyncTMAGatherOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand x_offsetsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> x_offsetsOperands(&x_offsetsRawOperand, 1);  ::llvm::SMLoc x_offsetsOperandsLoc;
  (void)x_offsetsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand y_offsetRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> y_offsetOperands(&y_offsetRawOperand, 1);  ::llvm::SMLoc y_offsetOperandsLoc;
  (void)y_offsetOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand resultRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> resultOperands(&resultRawOperand, 1);  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand barrierRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> barrierOperands(&barrierRawOperand, 1);  ::llvm::SMLoc barrierOperandsLoc;
  (void)barrierOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  x_offsetsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(x_offsetsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  y_offsetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(y_offsetRawOperand))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  resultOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(resultRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  barrierOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(barrierRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(desc_ptrOperands, x_offsetsOperands, y_offsetOperands, barrierOperands, resultOperands, predOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMAGatherOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getXOffsets();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getYOffset();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getResult();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBarrier();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
}

void AsyncTMAGatherOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(3);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(4);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMAGatherOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMAReduceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMAReduceOpGenericAdaptorBase::AsyncTMAReduceOpGenericAdaptorBase(AsyncTMAReduceOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AsyncTMAReduceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::triton::DescriptorReduceKind AsyncTMAReduceOpGenericAdaptorBase::getKind() {
  auto attr = getKindAttr();
  return attr.getValue();
}

} // namespace detail
AsyncTMAReduceOpAdaptor::AsyncTMAReduceOpAdaptor(AsyncTMAReduceOp op) : AsyncTMAReduceOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AsyncTMAReduceOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_kind = getProperties().kind; (void)tblgen_kind;
  if (!tblgen_kind) return emitError(loc, "'ttng.async_tma_reduce' op ""requires attribute 'kind'");

  if (tblgen_kind && !((::llvm::isa<::mlir::triton::DescriptorReduceKindAttr>(tblgen_kind))))
    return emitError(loc, "'ttng.async_tma_reduce' op ""attribute 'kind' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6, 7, 8");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsyncTMAReduceOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AsyncTMAReduceOp::getCoordMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AsyncTMAReduceOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.kind;
       auto attr = dict.get("kind");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `kind` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AsyncTMAReduceOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.kind;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("kind",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMAReduceOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.kind.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AsyncTMAReduceOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "kind")
      return prop.kind;
  return std::nullopt;
}

void AsyncTMAReduceOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "kind") {
       prop.kind = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.kind)>>(value);
       return;
    }
}

void AsyncTMAReduceOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.kind) attrs.append("kind", prop.kind);
}

::llvm::LogicalResult AsyncTMAReduceOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getKindAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(attr, "kind", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAReduceOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.kind)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMAReduceOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.kind);
}

::mlir::triton::DescriptorReduceKind AsyncTMAReduceOp::getKind() {
  auto attr = getKindAttr();
  return attr.getValue();
}

void AsyncTMAReduceOp::setKind(::mlir::triton::DescriptorReduceKind attrValue) {
  getProperties().kind = ::mlir::triton::DescriptorReduceKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::triton::DescriptorReduceKindAttr kind, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().kind = kind;
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::DescriptorReduceKindAttr kind, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().kind = kind;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::triton::DescriptorReduceKind kind, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().kind = ::mlir::triton::DescriptorReduceKindAttr::get(odsBuilder.getContext(), kind);
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::DescriptorReduceKind kind, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().kind = ::mlir::triton::DescriptorReduceKindAttr::get(odsBuilder.getContext(), kind);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AsyncTMAReduceOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AsyncTMAReduceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AsyncTMAReduceOp::verifyInvariantsImpl() {
  auto tblgen_kind = getProperties().kind; (void)tblgen_kind;
  if (!tblgen_kind) return emitOpError("requires attribute 'kind'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(*this, tblgen_kind, "kind")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAReduceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AsyncTMAReduceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::triton::DescriptorReduceKindAttr kindAttr;
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> coordOperands;
  ::llvm::SMLoc coordOperandsLoc;
  (void)coordOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type desc_ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> desc_ptrTypes(&desc_ptrRawType, 1);
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"add","min","max","inc","dec","and","or","xor"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "kind", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'kind' [add, min, max, inc, dec, and, or, xor]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeDescriptorReduceKind(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "kind attribute specification: \"" << attrStr << '"';;

      kindAttr = ::mlir::triton::DescriptorReduceKindAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<AsyncTMAReduceOp::Properties>().kind = kindAttr;
    }
  }
  if (parser.parseComma())
    return ::mlir::failure();

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  coordOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(coordOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(desc_ptrRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(srcRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(desc_ptrOperands, desc_ptrTypes, desc_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordOperands, odsBuildableType0, coordOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMAReduceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';

  {
    auto caseValue = getKind();
    auto caseValueStr = stringifyDescriptorReduceKind(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getCoord();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("kind");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getDescPtr().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getSrc().getType();
}

void AsyncTMAReduceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), 0, false, ::mlir::triton::GlobalMemory::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, ::mlir::triton::GlobalMemory::get());
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(2);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMAReduceOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMAScatterOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMAScatterOpGenericAdaptorBase::AsyncTMAScatterOpGenericAdaptorBase(AsyncTMAScatterOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

} // namespace detail
AsyncTMAScatterOpAdaptor::AsyncTMAScatterOpAdaptor(AsyncTMAScatterOp op) : AsyncTMAScatterOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AsyncTMAScatterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAScatterOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute AsyncTMAScatterOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMAScatterOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> AsyncTMAScatterOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void AsyncTMAScatterOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void AsyncTMAScatterOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult AsyncTMAScatterOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void AsyncTMAScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(x_offsets);
  odsState.addOperands(y_offset);
  odsState.addOperands(src);
}

void AsyncTMAScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(x_offsets);
  odsState.addOperands(y_offset);
  odsState.addOperands(src);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMAScatterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsyncTMAScatterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AsyncTMAScatterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AsyncTMAScatterOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AsyncTMAScatterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand x_offsetsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> x_offsetsOperands(&x_offsetsRawOperand, 1);  ::llvm::SMLoc x_offsetsOperandsLoc;
  (void)x_offsetsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand y_offsetRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> y_offsetOperands(&y_offsetRawOperand, 1);  ::llvm::SMLoc y_offsetOperandsLoc;
  (void)y_offsetOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  x_offsetsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(x_offsetsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  y_offsetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(y_offsetRawOperand))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(desc_ptrOperands, x_offsetsOperands, y_offsetOperands, srcOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMAScatterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getXOffsets();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getYOffset();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
}

void AsyncTMAScatterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::GlobalMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(3);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMAScatterOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::BarrierExpectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BarrierExpectOpGenericAdaptorBase::BarrierExpectOpGenericAdaptorBase(BarrierExpectOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t BarrierExpectOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
BarrierExpectOpAdaptor::BarrierExpectOpAdaptor(BarrierExpectOp op) : BarrierExpectOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult BarrierExpectOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitError(loc, "'ttng.barrier_expect' op ""requires attribute 'size'");

  if (tblgen_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_size).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.barrier_expect' op ""attribute 'size' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult BarrierExpectOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.size;
       auto attr = dict.get("size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute BarrierExpectOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("size",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code BarrierExpectOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.size.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> BarrierExpectOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "size")
      return prop.size;
  return std::nullopt;
}

void BarrierExpectOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "size") {
       prop.size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.size)>>(value);
       return;
    }
}

void BarrierExpectOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.size) attrs.append("size", prop.size);
}

::llvm::LogicalResult BarrierExpectOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "size", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult BarrierExpectOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.size)))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierExpectOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.size);
}

uint32_t BarrierExpectOp::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

void BarrierExpectOp::setSize(uint32_t attrValue) {
  getProperties().size = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = size;
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = size;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierExpectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<BarrierExpectOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void BarrierExpectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult BarrierExpectOp::verifyInvariantsImpl() {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitOpError("requires attribute 'size'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_size, "size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult BarrierExpectOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BarrierExpectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::IntegerAttr sizeAttr;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(sizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (sizeAttr) result.getOrAddProperties<BarrierExpectOp::Properties>().size = sizeAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(allocRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType0, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierExpectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getSizeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("size");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getAlloc().getType();
}

void BarrierExpectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::BarrierExpectOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterArriveOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ClusterArriveOpGenericAdaptorBase::ClusterArriveOpGenericAdaptorBase(ClusterArriveOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

bool ClusterArriveOpGenericAdaptorBase::getRelaxed() {
  auto attr = getRelaxedAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ClusterArriveOpAdaptor::ClusterArriveOpAdaptor(ClusterArriveOp op) : ClusterArriveOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ClusterArriveOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_relaxed = getProperties().relaxed; (void)tblgen_relaxed;
  if (!tblgen_relaxed) return emitError(loc, "'ttng.cluster_arrive' op ""requires attribute 'relaxed'");

  if (tblgen_relaxed && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_relaxed))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_relaxed).getType().isSignlessInteger(1)))))
    return emitError(loc, "'ttng.cluster_arrive' op ""attribute 'relaxed' failed to satisfy constraint: 1-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ClusterArriveOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.relaxed;
       auto attr = dict.get("relaxed");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `relaxed` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ClusterArriveOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.relaxed;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("relaxed",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ClusterArriveOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.relaxed.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ClusterArriveOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "relaxed")
      return prop.relaxed;
  return std::nullopt;
}

void ClusterArriveOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "relaxed") {
       prop.relaxed = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.relaxed)>>(value);
       return;
    }
}

void ClusterArriveOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.relaxed) attrs.append("relaxed", prop.relaxed);
}

::llvm::LogicalResult ClusterArriveOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getRelaxedAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps6(attr, "relaxed", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ClusterArriveOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.relaxed)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ClusterArriveOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.relaxed);
}

bool ClusterArriveOp::getRelaxed() {
  auto attr = getRelaxedAttr();
  return attr.getValue().getZExtValue();
}

void ClusterArriveOp::setRelaxed(bool attrValue) {
  getProperties().relaxed = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(1), attrValue);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = relaxed;
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = relaxed;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), relaxed);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), relaxed);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterArriveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ClusterArriveOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ClusterArriveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ClusterArriveOp::verifyInvariantsImpl() {
  auto tblgen_relaxed = getProperties().relaxed; (void)tblgen_relaxed;
  if (!tblgen_relaxed) return emitOpError("requires attribute 'relaxed'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps6(*this, tblgen_relaxed, "relaxed")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult ClusterArriveOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ClusterArriveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ClusterArriveOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterArriveOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ClusterWaitOpGenericAdaptorBase::ClusterWaitOpGenericAdaptorBase(ClusterWaitOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

} // namespace detail
ClusterWaitOpAdaptor::ClusterWaitOpAdaptor(ClusterWaitOp op) : ClusterWaitOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ClusterWaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ClusterWaitOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ClusterWaitOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ClusterWaitOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ClusterWaitOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ClusterWaitOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ClusterWaitOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ClusterWaitOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ClusterWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ClusterWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ClusterWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ClusterWaitOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult ClusterWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ClusterWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ClusterWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::FenceAsyncSharedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FenceAsyncSharedOpGenericAdaptorBase::FenceAsyncSharedOpGenericAdaptorBase(FenceAsyncSharedOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

bool FenceAsyncSharedOpGenericAdaptorBase::getBCluster() {
  auto attr = getBClusterAttr();
  return attr.getValue();
}

} // namespace detail
FenceAsyncSharedOpAdaptor::FenceAsyncSharedOpAdaptor(FenceAsyncSharedOp op) : FenceAsyncSharedOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FenceAsyncSharedOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_bCluster = getProperties().bCluster; (void)tblgen_bCluster;
  if (!tblgen_bCluster) return emitError(loc, "'ttng.fence_async_shared' op ""requires attribute 'bCluster'");

  if (tblgen_bCluster && !((::llvm::isa<::mlir::BoolAttr>(tblgen_bCluster))))
    return emitError(loc, "'ttng.fence_async_shared' op ""attribute 'bCluster' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::llvm::LogicalResult FenceAsyncSharedOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.bCluster;
       auto attr = dict.get("bCluster");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bCluster` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FenceAsyncSharedOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.bCluster;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bCluster",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FenceAsyncSharedOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.bCluster.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FenceAsyncSharedOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "bCluster")
      return prop.bCluster;
  return std::nullopt;
}

void FenceAsyncSharedOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "bCluster") {
       prop.bCluster = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bCluster)>>(value);
       return;
    }
}

void FenceAsyncSharedOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.bCluster) attrs.append("bCluster", prop.bCluster);
}

::llvm::LogicalResult FenceAsyncSharedOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBClusterAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, "bCluster", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult FenceAsyncSharedOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.bCluster)))
    return ::mlir::failure();
  return ::mlir::success();
}

void FenceAsyncSharedOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.bCluster);
}

bool FenceAsyncSharedOp::getBCluster() {
  auto attr = getBClusterAttr();
  return attr.getValue();
}

void FenceAsyncSharedOp::setBCluster(bool attrValue) {
  getProperties().bCluster = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::BoolAttr bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = bCluster;
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::BoolAttr bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = bCluster;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = odsBuilder.getBoolAttr(bCluster);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = odsBuilder.getBoolAttr(bCluster);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<FenceAsyncSharedOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult FenceAsyncSharedOp::verifyInvariantsImpl() {
  auto tblgen_bCluster = getProperties().bCluster; (void)tblgen_bCluster;
  if (!tblgen_bCluster) return emitOpError("requires attribute 'bCluster'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(*this, tblgen_bCluster, "bCluster")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult FenceAsyncSharedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FenceAsyncSharedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void FenceAsyncSharedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::FenceAsyncSharedOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InitBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InitBarrierOpGenericAdaptorBase::InitBarrierOpGenericAdaptorBase(InitBarrierOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t InitBarrierOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
InitBarrierOpAdaptor::InitBarrierOpAdaptor(InitBarrierOp op) : InitBarrierOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult InitBarrierOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'ttng.init_barrier' op ""requires attribute 'count'");

  if (tblgen_count && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.init_barrier' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult InitBarrierOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.count;
       auto attr = dict.get("count");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `count` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute InitBarrierOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code InitBarrierOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> InitBarrierOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void InitBarrierOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "count") {
       prop.count = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void InitBarrierOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.count) attrs.append("count", prop.count);
}

::llvm::LogicalResult InitBarrierOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "count", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult InitBarrierOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.count)))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitBarrierOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.count);
}

uint32_t InitBarrierOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

void InitBarrierOp::setCount(uint32_t attrValue) {
  getProperties().count = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = count;
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = count;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<InitBarrierOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void InitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult InitBarrierOp::verifyInvariantsImpl() {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_count, "count")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult InitBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InitBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<InitBarrierOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(allocRawType))
    return ::mlir::failure();
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("count");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getAlloc().getType();
}

void InitBarrierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InitBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InvalBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InvalBarrierOpGenericAdaptorBase::InvalBarrierOpGenericAdaptorBase(InvalBarrierOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

} // namespace detail
InvalBarrierOpAdaptor::InvalBarrierOpAdaptor(InvalBarrierOp op) : InvalBarrierOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult InvalBarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult InvalBarrierOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute InvalBarrierOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code InvalBarrierOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> InvalBarrierOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void InvalBarrierOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void InvalBarrierOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult InvalBarrierOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void InvalBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc) {
  odsState.addOperands(alloc);
}

void InvalBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc) {
  odsState.addOperands(alloc);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InvalBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InvalBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult InvalBarrierOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult InvalBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InvalBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(allocRawType))
    return ::mlir::failure();
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InvalBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getAlloc().getType();
}

void InvalBarrierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InvalBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TCGen5MMAOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TCGen5MMAOpGenericAdaptorBase::TCGen5MMAOpGenericAdaptorBase(TCGen5MMAOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TCGen5MMAOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::std::optional<bool> TCGen5MMAOpGenericAdaptorBase::getTwoCtas() {
  auto attr = getTwoCtasAttr();
  return attr ? ::std::optional<bool>(attr != nullptr) : (::std::nullopt);
}

} // namespace detail
TCGen5MMAOpAdaptor::TCGen5MMAOpAdaptor(TCGen5MMAOp op) : TCGen5MMAOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TCGen5MMAOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_two_ctas = getProperties().two_ctas; (void)tblgen_two_ctas;

  if (tblgen_two_ctas && !((::llvm::isa<::mlir::UnitAttr>(tblgen_two_ctas))))
    return emitError(loc, "'ttng.tc_gen5_mma' op ""attribute 'two_ctas' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TCGen5MMAOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange TCGen5MMAOp::getAccDepMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange TCGen5MMAOp::getBarriersMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange TCGen5MMAOp::getBarrierPredsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> TCGen5MMAOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult TCGen5MMAOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.two_ctas;
       auto attr = dict.get("two_ctas");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `two_ctas` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) -> ::mlir::LogicalResult {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute TCGen5MMAOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.two_ctas;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("two_ctas",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      auto attr = [&]() -> ::mlir::Attribute {
        return ::mlir::DenseI32ArrayAttr::get(ctx, propStorage);
      }();
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes", attr));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TCGen5MMAOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    using ::llvm::hash_value;
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.two_ctas.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> TCGen5MMAOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "two_ctas")
      return prop.two_ctas;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void TCGen5MMAOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "two_ctas") {
       prop.two_ctas = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.two_ctas)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void TCGen5MMAOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.two_ctas) attrs.append("two_ctas", prop.two_ctas);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult TCGen5MMAOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTwoCtasAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps7(attr, "two_ctas", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TCGen5MMAOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  if (::mlir::failed(reader.readOptionalAttribute(prop.two_ctas)))
    return ::mlir::failure();

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void TCGen5MMAOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  writer.writeOptionalAttribute(prop.two_ctas);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::std::optional<bool> TCGen5MMAOp::getTwoCtas() {
  auto attr = getTwoCtasAttr();
  return attr ? ::std::optional<bool>(attr != nullptr) : (::std::nullopt);
}

void TCGen5MMAOp::setTwoCtas(bool attrValue) {
    auto &odsProp = getProperties().two_ctas;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void TCGen5MMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type token, ::mlir::Value a, ::mlir::Value b, ::mlir::Value d, /*optional*/::mlir::Value acc_dep, ::mlir::Value useD, ::mlir::Value pred, ::mlir::ValueRange barriers, ::mlir::ValueRange barrier_preds, /*optional*/::mlir::UnitAttr two_ctas) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(d);
  if (acc_dep)
    odsState.addOperands(acc_dep);
  odsState.addOperands(useD);
  odsState.addOperands(pred);
  odsState.addOperands(barriers);
  odsState.addOperands(barrier_preds);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, (acc_dep ? 1 : 0), 1, 1, static_cast<int32_t>(barriers.size()), static_cast<int32_t>(barrier_preds.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (two_ctas) {
    odsState.getOrAddProperties<Properties>().two_ctas = two_ctas;
  }
  if (token)
    odsState.addTypes(token);
}

void TCGen5MMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value d, /*optional*/::mlir::Value acc_dep, ::mlir::Value useD, ::mlir::Value pred, ::mlir::ValueRange barriers, ::mlir::ValueRange barrier_preds, /*optional*/::mlir::UnitAttr two_ctas) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(d);
  if (acc_dep)
    odsState.addOperands(acc_dep);
  odsState.addOperands(useD);
  odsState.addOperands(pred);
  odsState.addOperands(barriers);
  odsState.addOperands(barrier_preds);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, (acc_dep ? 1 : 0), 1, 1, static_cast<int32_t>(barriers.size()), static_cast<int32_t>(barrier_preds.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (two_ctas) {
    odsState.getOrAddProperties<Properties>().two_ctas = two_ctas;
  }
  odsState.addTypes(resultTypes);
}

void TCGen5MMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TCGen5MMAOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TCGen5MMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TCGen5MMAOp::verifyInvariantsImpl() {
  auto tblgen_two_ctas = getProperties().two_ctas; (void)tblgen_two_ctas;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps7(*this, tblgen_two_ctas, "two_ctas")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TCGen5MMAOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TCGen5MMAOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dOperands(&dRawOperand, 1);  ::llvm::SMLoc dOperandsLoc;
  (void)dOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> acc_depOperands;
  ::llvm::SMLoc acc_depOperandsLoc;
  (void)acc_depOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> tokenTypes;
  ::mlir::OpAsmParser::UnresolvedOperand useDRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> useDOperands(&useDRawOperand, 1);  ::llvm::SMLoc useDOperandsLoc;
  (void)useDOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> barriersOperands;
  ::llvm::SMLoc barriersOperandsLoc;
  (void)barriersOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> barrier_predsOperands;
  ::llvm::SMLoc barrier_predsOperandsLoc;
  (void)barrier_predsOperandsLoc;
  ::mlir::Type aRawType{};
  ::llvm::ArrayRef<::mlir::Type> aTypes(&aRawType, 1);
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type dRawType{};
  ::llvm::ArrayRef<::mlir::Type> dTypes(&dRawType, 1);
  ::llvm::SmallVector<::mlir::Type, 1> barriersTypes;

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dRawOperand))
    return ::mlir::failure();
  {
    acc_depOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> acc_depOperand;
    ::mlir::Type tokenType;
    auto odsResult = parseToken(parser, acc_depOperand, tokenType);
    if (odsResult) return ::mlir::failure();
    if (acc_depOperand.has_value())
      acc_depOperands.push_back(*acc_depOperand);
    if (tokenType)
      tokenTypes.push_back(tokenType);
  }
  if (parser.parseComma())
    return ::mlir::failure();

  useDOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(useDRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  {
    barriersOperandsLoc = parser.getCurrentLocation();
    barrier_predsOperandsLoc = parser.getCurrentLocation();
    auto odsResult = parseBarriersAndPreds(parser, barriersOperands, barrier_predsOperands);
    if (odsResult) return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(aRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(bRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(dRawType))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(barriersTypes))
    return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, static_cast<int32_t>(acc_depOperands.size()), 1, 1, static_cast<int32_t>(barriersOperands.size()), static_cast<int32_t>(barrier_predsOperands.size())}), result.getOrAddProperties<TCGen5MMAOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::triton::gpu::AsyncTokenType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(tokenTypes);
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dOperands, dTypes, dOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(acc_depOperands, odsBuildableType0, acc_depOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(useDOperands, odsBuildableType1, useDOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barriersOperands, barriersTypes, barriersOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barrier_predsOperands, odsBuildableType1, barrier_predsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TCGen5MMAOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getD();
  printToken(_odsPrinter, *this, getAccDep(), (getToken() ? getToken().getType() : ::mlir::Type()));
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getUseD();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  printBarriersAndPreds(_odsPrinter, *this, getBarriers(), getBarrierPreds());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getA().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getB().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getD().getType();
  if (!getBarriers().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getBarriers().getTypes();
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TCGen5MMAOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TCGen5MMAScaledOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TCGen5MMAScaledOpGenericAdaptorBase::TCGen5MMAScaledOpGenericAdaptorBase(TCGen5MMAScaledOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TCGen5MMAScaledOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::triton::ScaleDotElemType TCGen5MMAScaledOpGenericAdaptorBase::getAType() {
  auto attr = getATypeAttr();
  return attr.getValue();
}

::mlir::triton::ScaleDotElemType TCGen5MMAScaledOpGenericAdaptorBase::getBType() {
  auto attr = getBTypeAttr();
  return attr.getValue();
}

} // namespace detail
TCGen5MMAScaledOpAdaptor::TCGen5MMAScaledOpAdaptor(TCGen5MMAScaledOp op) : TCGen5MMAScaledOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TCGen5MMAScaledOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_a_type = getProperties().a_type; (void)tblgen_a_type;
  if (!tblgen_a_type) return emitError(loc, "'ttng.tc_gen5_mma_scaled' op ""requires attribute 'a_type'");
  auto tblgen_b_type = getProperties().b_type; (void)tblgen_b_type;
  if (!tblgen_b_type) return emitError(loc, "'ttng.tc_gen5_mma_scaled' op ""requires attribute 'b_type'");

  if (tblgen_a_type && !((::llvm::isa<::mlir::triton::ScaleDotElemTypeAttr>(tblgen_a_type))))
    return emitError(loc, "'ttng.tc_gen5_mma_scaled' op ""attribute 'a_type' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6");

  if (tblgen_b_type && !((::llvm::isa<::mlir::triton::ScaleDotElemTypeAttr>(tblgen_b_type))))
    return emitError(loc, "'ttng.tc_gen5_mma_scaled' op ""attribute 'b_type' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TCGen5MMAScaledOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange TCGen5MMAScaledOp::getAccDepMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange TCGen5MMAScaledOp::getBarriersMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange TCGen5MMAScaledOp::getBarrierPredsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> TCGen5MMAScaledOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult TCGen5MMAScaledOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.a_type;
       auto attr = dict.get("a_type");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `a_type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.b_type;
       auto attr = dict.get("b_type");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `b_type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) -> ::mlir::LogicalResult {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute TCGen5MMAScaledOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.a_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("a_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.b_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("b_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      auto attr = [&]() -> ::mlir::Attribute {
        return ::mlir::DenseI32ArrayAttr::get(ctx, propStorage);
      }();
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes", attr));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TCGen5MMAScaledOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    using ::llvm::hash_value;
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.a_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.b_type.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> TCGen5MMAScaledOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "a_type")
      return prop.a_type;

    if (name == "b_type")
      return prop.b_type;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void TCGen5MMAScaledOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "a_type") {
       prop.a_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.a_type)>>(value);
       return;
    }

    if (name == "b_type") {
       prop.b_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.b_type)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void TCGen5MMAScaledOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.a_type) attrs.append("a_type", prop.a_type);

    if (prop.b_type) attrs.append("b_type", prop.b_type);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult TCGen5MMAScaledOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getATypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(attr, "a_type", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(attr, "b_type", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TCGen5MMAScaledOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.a_type)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.b_type)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void TCGen5MMAScaledOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.a_type);
  writer.writeAttribute(prop.b_type);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::mlir::triton::ScaleDotElemType TCGen5MMAScaledOp::getAType() {
  auto attr = getATypeAttr();
  return attr.getValue();
}

::mlir::triton::ScaleDotElemType TCGen5MMAScaledOp::getBType() {
  auto attr = getBTypeAttr();
  return attr.getValue();
}

void TCGen5MMAScaledOp::setAType(::mlir::triton::ScaleDotElemType attrValue) {
  getProperties().a_type = ::mlir::triton::ScaleDotElemTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void TCGen5MMAScaledOp::setBType(::mlir::triton::ScaleDotElemType attrValue) {
  getProperties().b_type = ::mlir::triton::ScaleDotElemTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void TCGen5MMAScaledOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type token, ::mlir::Value a, ::mlir::Value b, ::mlir::Value d, /*optional*/::mlir::Value acc_dep, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::triton::ScaleDotElemTypeAttr a_type, ::mlir::triton::ScaleDotElemTypeAttr b_type, ::mlir::Value useD, ::mlir::Value pred, ::mlir::ValueRange barriers, ::mlir::ValueRange barrier_preds) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(d);
  if (acc_dep)
    odsState.addOperands(acc_dep);
  odsState.addOperands(a_scale);
  odsState.addOperands(b_scale);
  odsState.addOperands(useD);
  odsState.addOperands(pred);
  odsState.addOperands(barriers);
  odsState.addOperands(barrier_preds);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, (acc_dep ? 1 : 0), 1, 1, 1, 1, static_cast<int32_t>(barriers.size()), static_cast<int32_t>(barrier_preds.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().a_type = a_type;
  odsState.getOrAddProperties<Properties>().b_type = b_type;
  if (token)
    odsState.addTypes(token);
}

void TCGen5MMAScaledOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value d, /*optional*/::mlir::Value acc_dep, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::triton::ScaleDotElemTypeAttr a_type, ::mlir::triton::ScaleDotElemTypeAttr b_type, ::mlir::Value useD, ::mlir::Value pred, ::mlir::ValueRange barriers, ::mlir::ValueRange barrier_preds) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(d);
  if (acc_dep)
    odsState.addOperands(acc_dep);
  odsState.addOperands(a_scale);
  odsState.addOperands(b_scale);
  odsState.addOperands(useD);
  odsState.addOperands(pred);
  odsState.addOperands(barriers);
  odsState.addOperands(barrier_preds);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, (acc_dep ? 1 : 0), 1, 1, 1, 1, static_cast<int32_t>(barriers.size()), static_cast<int32_t>(barrier_preds.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().a_type = a_type;
  odsState.getOrAddProperties<Properties>().b_type = b_type;
  odsState.addTypes(resultTypes);
}

void TCGen5MMAScaledOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value d, /*optional*/::mlir::Value acc_dep, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::triton::ScaleDotElemType a_type, ::mlir::triton::ScaleDotElemType b_type, ::mlir::Value useD, ::mlir::Value pred, ::mlir::ValueRange barriers, ::mlir::ValueRange barrier_preds) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(d);
  if (acc_dep)
    odsState.addOperands(acc_dep);
  odsState.addOperands(a_scale);
  odsState.addOperands(b_scale);
  odsState.addOperands(useD);
  odsState.addOperands(pred);
  odsState.addOperands(barriers);
  odsState.addOperands(barrier_preds);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, (acc_dep ? 1 : 0), 1, 1, 1, 1, static_cast<int32_t>(barriers.size()), static_cast<int32_t>(barrier_preds.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().a_type = ::mlir::triton::ScaleDotElemTypeAttr::get(odsBuilder.getContext(), a_type);
  odsState.getOrAddProperties<Properties>().b_type = ::mlir::triton::ScaleDotElemTypeAttr::get(odsBuilder.getContext(), b_type);
  odsState.addTypes(resultTypes);
}

void TCGen5MMAScaledOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 7u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TCGen5MMAScaledOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TCGen5MMAScaledOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 7u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TCGen5MMAScaledOp::verifyInvariantsImpl() {
  auto tblgen_a_type = getProperties().a_type; (void)tblgen_a_type;
  if (!tblgen_a_type) return emitOpError("requires attribute 'a_type'");
  auto tblgen_b_type = getProperties().b_type; (void)tblgen_b_type;
  if (!tblgen_b_type) return emitOpError("requires attribute 'b_type'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(*this, tblgen_a_type, "a_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps8(*this, tblgen_b_type, "b_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TCGen5MMAScaledOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TCGen5MMAScaledOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dOperands(&dRawOperand, 1);  ::llvm::SMLoc dOperandsLoc;
  (void)dOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> acc_depOperands;
  ::llvm::SMLoc acc_depOperandsLoc;
  (void)acc_depOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> tokenTypes;
  ::mlir::OpAsmParser::UnresolvedOperand a_scaleRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> a_scaleOperands(&a_scaleRawOperand, 1);  ::llvm::SMLoc a_scaleOperandsLoc;
  (void)a_scaleOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand b_scaleRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> b_scaleOperands(&b_scaleRawOperand, 1);  ::llvm::SMLoc b_scaleOperandsLoc;
  (void)b_scaleOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand useDRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> useDOperands(&useDRawOperand, 1);  ::llvm::SMLoc useDOperandsLoc;
  (void)useDOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::triton::ScaleDotElemTypeAttr a_typeAttr;
  ::mlir::triton::ScaleDotElemTypeAttr b_typeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> barriersOperands;
  ::llvm::SMLoc barriersOperandsLoc;
  (void)barriersOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> barrier_predsOperands;
  ::llvm::SMLoc barrier_predsOperandsLoc;
  (void)barrier_predsOperandsLoc;
  ::mlir::Type aRawType{};
  ::llvm::ArrayRef<::mlir::Type> aTypes(&aRawType, 1);
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type dRawType{};
  ::llvm::ArrayRef<::mlir::Type> dTypes(&dRawType, 1);
  ::mlir::Type a_scaleRawType{};
  ::llvm::ArrayRef<::mlir::Type> a_scaleTypes(&a_scaleRawType, 1);
  ::mlir::Type b_scaleRawType{};
  ::llvm::ArrayRef<::mlir::Type> b_scaleTypes(&b_scaleRawType, 1);
  ::llvm::SmallVector<::mlir::Type, 1> barriersTypes;

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dRawOperand))
    return ::mlir::failure();
  {
    acc_depOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> acc_depOperand;
    ::mlir::Type tokenType;
    auto odsResult = parseToken(parser, acc_depOperand, tokenType);
    if (odsResult) return ::mlir::failure();
    if (acc_depOperand.has_value())
      acc_depOperands.push_back(*acc_depOperand);
    if (tokenType)
      tokenTypes.push_back(tokenType);
  }
  if (parser.parseComma())
    return ::mlir::failure();

  a_scaleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(a_scaleRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  b_scaleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(b_scaleRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  useDOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(useDRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("lhs"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"e4m3","e5m2","e2m3","e3m2","e2m1","bf16","fp16"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "a_type", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'a_type' [e4m3, e5m2, e2m3, e3m2, e2m1, bf16, fp16]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeScaleDotElemType(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "a_type attribute specification: \"" << attrStr << '"';;

      a_typeAttr = ::mlir::triton::ScaleDotElemTypeAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<TCGen5MMAScaledOp::Properties>().a_type = a_typeAttr;
    }
  }
  if (parser.parseKeyword("rhs"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"e4m3","e5m2","e2m3","e3m2","e2m1","bf16","fp16"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "b_type", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'b_type' [e4m3, e5m2, e2m3, e3m2, e2m1, bf16, fp16]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeScaleDotElemType(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "b_type attribute specification: \"" << attrStr << '"';;

      b_typeAttr = ::mlir::triton::ScaleDotElemTypeAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<TCGen5MMAScaledOp::Properties>().b_type = b_typeAttr;
    }
  }
  {
    barriersOperandsLoc = parser.getCurrentLocation();
    barrier_predsOperandsLoc = parser.getCurrentLocation();
    auto odsResult = parseBarriersAndPreds(parser, barriersOperands, barrier_predsOperands);
    if (odsResult) return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(aRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(bRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(dRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(a_scaleRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(b_scaleRawType))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(barriersTypes))
    return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, static_cast<int32_t>(acc_depOperands.size()), 1, 1, 1, 1, static_cast<int32_t>(barriersOperands.size()), static_cast<int32_t>(barrier_predsOperands.size())}), result.getOrAddProperties<TCGen5MMAScaledOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::triton::gpu::AsyncTokenType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(tokenTypes);
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dOperands, dTypes, dOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(acc_depOperands, odsBuildableType0, acc_depOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(a_scaleOperands, a_scaleTypes, a_scaleOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(b_scaleOperands, b_scaleTypes, b_scaleOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(useDOperands, odsBuildableType1, useDOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barriersOperands, barriersTypes, barriersOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barrier_predsOperands, odsBuildableType1, barrier_predsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TCGen5MMAScaledOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getD();
  printToken(_odsPrinter, *this, getAccDep(), (getToken() ? getToken().getType() : ::mlir::Type()));
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAScale();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBScale();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getUseD();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  _odsPrinter << ' ' << "lhs";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getAType();
    auto caseValueStr = stringifyScaleDotElemType(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ' ' << "rhs";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getBType();
    auto caseValueStr = stringifyScaleDotElemType(caseValue);
    _odsPrinter << caseValueStr;
  }
  printBarriersAndPreds(_odsPrinter, *this, getBarriers(), getBarrierPreds());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("a_type");
  elidedAttrs.push_back("b_type");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getA().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getB().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getD().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getAScale().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getBScale().getType();
  if (!getBarriers().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getBarriers().getTypes();
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TCGen5MMAScaledOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMAStoreWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMAStoreWaitOpGenericAdaptorBase::TMAStoreWaitOpGenericAdaptorBase(TMAStoreWaitOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t TMAStoreWaitOpGenericAdaptorBase::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TMAStoreWaitOpAdaptor::TMAStoreWaitOpAdaptor(TMAStoreWaitOp op) : TMAStoreWaitOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMAStoreWaitOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitError(loc, "'ttng.async_tma_store_wait' op ""requires attribute 'pendings'");

  if (tblgen_pendings && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pendings))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pendings).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.async_tma_store_wait' op ""attribute 'pendings' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TMAStoreWaitOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.pendings;
       auto attr = dict.get("pendings");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pendings` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TMAStoreWaitOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.pendings;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pendings",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMAStoreWaitOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.pendings.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TMAStoreWaitOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "pendings")
      return prop.pendings;
  return std::nullopt;
}

void TMAStoreWaitOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "pendings") {
       prop.pendings = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pendings)>>(value);
       return;
    }
}

void TMAStoreWaitOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.pendings) attrs.append("pendings", prop.pendings);
}

::llvm::LogicalResult TMAStoreWaitOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getPendingsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "pendings", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TMAStoreWaitOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.pendings)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMAStoreWaitOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.pendings);
}

uint32_t TMAStoreWaitOp::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

void TMAStoreWaitOp::setPendings(uint32_t attrValue) {
  getProperties().pendings = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr pendings) {
  odsState.getOrAddProperties<Properties>().pendings = pendings;
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr pendings) {
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t pendings) {
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t pendings) {
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TMAStoreWaitOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TMAStoreWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMAStoreWaitOp::verifyInvariantsImpl() {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitOpError("requires attribute 'pendings'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_pendings, "pendings")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult TMAStoreWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TMAStoreWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void TMAStoreWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMAStoreWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMEMAllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMEMAllocOpGenericAdaptorBase::TMEMAllocOpGenericAdaptorBase(TMEMAllocOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TMEMAllocOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
TMEMAllocOpAdaptor::TMEMAllocOpAdaptor(TMEMAllocOp op) : TMEMAllocOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMEMAllocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void TMEMAllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "result");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "token");
}

std::pair<unsigned, unsigned> TMEMAllocOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange TMEMAllocOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TMEMAllocOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult TMEMAllocOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TMEMAllocOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMEMAllocOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TMEMAllocOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TMEMAllocOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TMEMAllocOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TMEMAllocOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TMEMAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Type token, /*optional*/::mlir::Value src) {
  if (src)
    odsState.addOperands(src);
  odsState.addTypes(result);
  if (token)
    odsState.addTypes(token);
}

void TMEMAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value src) {
  if (src)
    odsState.addOperands(src);
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMEMAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TMEMAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMEMAllocOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TMEMAllocOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TMEMAllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> srcOperands;
  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  {
    srcOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      srcOperands.push_back(operand);
    }
  }
  if (!srcOperands.empty()) {
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(srcOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMAllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getSrc()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getSrc())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMAllocOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMEMCopyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMEMCopyOpGenericAdaptorBase::TMEMCopyOpGenericAdaptorBase(TMEMCopyOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TMEMCopyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
TMEMCopyOpAdaptor::TMEMCopyOpAdaptor(TMEMCopyOp op) : TMEMCopyOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMEMCopyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TMEMCopyOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange TMEMCopyOp::getBarrierMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult TMEMCopyOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TMEMCopyOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMEMCopyOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TMEMCopyOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TMEMCopyOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TMEMCopyOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TMEMCopyOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TMEMCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, /*optional*/::mlir::Value barrier) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
  if (barrier)
    odsState.addOperands(barrier);
}

void TMEMCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, /*optional*/::mlir::Value barrier) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
  if (barrier)
    odsState.addOperands(barrier);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMEMCopyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TMEMCopyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMEMCopyOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TMEMCopyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TMEMCopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(&dstRawOperand, 1);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> barrierOperands;
  ::llvm::SMLoc barrierOperandsLoc;
  (void)barrierOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  {
    barrierOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      barrierOperands.push_back(operand);
    }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(srcOperands, dstOperands, barrierOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMCopyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDst();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  if (::mlir::Value value = getBarrier())
    _odsPrinter << value;
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void TMEMCopyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(1);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::nvidia_gpu::TensorMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMCopyOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMEMLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMEMLoadOpGenericAdaptorBase::TMEMLoadOpGenericAdaptorBase(TMEMLoadOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TMEMLoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
TMEMLoadOpAdaptor::TMEMLoadOpAdaptor(TMEMLoadOp op) : TMEMLoadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMEMLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void TMEMLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "result");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "token");
}

std::pair<unsigned, unsigned> TMEMLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange TMEMLoadOp::getDepMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TMEMLoadOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult TMEMLoadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TMEMLoadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMEMLoadOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TMEMLoadOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TMEMLoadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TMEMLoadOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TMEMLoadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TMEMLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Type token, ::mlir::Value src, /*optional*/::mlir::Value dep) {
  odsState.addOperands(src);
  if (dep)
    odsState.addOperands(dep);
  odsState.addTypes(result);
  if (token)
    odsState.addTypes(token);
}

void TMEMLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, /*optional*/::mlir::Value dep) {
  odsState.addOperands(src);
  if (dep)
    odsState.addOperands(dep);
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMEMLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TMEMLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMEMLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TMEMLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TMEMLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depOperands;
  ::llvm::SMLoc depOperandsLoc;
  (void)depOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> tokenTypes;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    depOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> depOperand;
    ::mlir::Type tokenType;
    auto odsResult = parseToken(parser, depOperand, tokenType);
    if (odsResult) return ::mlir::failure();
    if (depOperand.has_value())
      depOperands.push_back(*depOperand);
    if (tokenType)
      tokenTypes.push_back(tokenType);
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(srcRawType))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::triton::gpu::AsyncTokenType>();
  result.addTypes(resultTypes);
  result.addTypes(tokenTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(depOperands, odsBuildableType0, depOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  printToken(_odsPrinter, *this, getDep(), (getToken() ? getToken().getType() : ::mlir::Type()));
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getSrc().getType();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TMEMLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::nvidia_gpu::TensorMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMLoadOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMEMStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMEMStoreOpGenericAdaptorBase::TMEMStoreOpGenericAdaptorBase(TMEMStoreOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TMEMStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
TMEMStoreOpAdaptor::TMEMStoreOpAdaptor(TMEMStoreOp op) : TMEMStoreOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMEMStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TMEMStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange TMEMStoreOp::getDepMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TMEMStoreOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult TMEMStoreOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TMEMStoreOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMEMStoreOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TMEMStoreOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TMEMStoreOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TMEMStoreOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TMEMStoreOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TMEMStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value dst, Value src, Value pred) {
      build(odsBuilder, odsState, Type(), dst, Value(), src, pred);
    
}

void TMEMStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type token, ::mlir::Value dst, /*optional*/::mlir::Value dep, ::mlir::Value src, ::mlir::Value pred) {
  odsState.addOperands(dst);
  if (dep)
    odsState.addOperands(dep);
  odsState.addOperands(src);
  odsState.addOperands(pred);
  if (token)
    odsState.addTypes(token);
}

void TMEMStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, /*optional*/::mlir::Value dep, ::mlir::Value src, ::mlir::Value pred) {
  odsState.addOperands(dst);
  if (dep)
    odsState.addOperands(dep);
  odsState.addOperands(src);
  odsState.addOperands(pred);
  odsState.addTypes(resultTypes);
}

void TMEMStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

void TMEMStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMEMStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TMEMStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TMEMStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(&dstRawOperand, 1);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depOperands;
  ::llvm::SMLoc depOperandsLoc;
  (void)depOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> tokenTypes;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type dstRawType{};
  ::llvm::ArrayRef<::mlir::Type> dstTypes(&dstRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperand))
    return ::mlir::failure();
  {
    depOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> depOperand;
    ::mlir::Type tokenType;
    auto odsResult = parseToken(parser, depOperand, tokenType);
    if (odsResult) return ::mlir::failure();
    if (depOperand.has_value())
      depOperands.push_back(*depOperand);
    if (tokenType)
      tokenTypes.push_back(tokenType);
  }
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(dstRawType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::triton::gpu::AsyncTokenType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(tokenTypes);
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(depOperands, odsBuildableType0, depOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDst();
  printToken(_odsPrinter, *this, getDep(), (getToken() ? getToken().getType() : ::mlir::Type()));
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
   _odsPrinter << getDst().getType();
}

void TMEMStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::nvidia_gpu::TensorMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMStoreOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMEMSubSliceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMEMSubSliceOpGenericAdaptorBase::TMEMSubSliceOpGenericAdaptorBase(TMEMSubSliceOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t TMEMSubSliceOpGenericAdaptorBase::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TMEMSubSliceOpAdaptor::TMEMSubSliceOpAdaptor(TMEMSubSliceOp op) : TMEMSubSliceOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TMEMSubSliceOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_N = getProperties().N; (void)tblgen_N;
  if (!tblgen_N) return emitError(loc, "'ttng.tmem_subslice' op ""requires attribute 'N'");

  if (tblgen_N && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_N))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_N).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.tmem_subslice' op ""attribute 'N' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TMEMSubSliceOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.N;
       auto attr = dict.get("N");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `N` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TMEMSubSliceOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.N;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("N",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMEMSubSliceOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.N.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TMEMSubSliceOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "N")
      return prop.N;
  return std::nullopt;
}

void TMEMSubSliceOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "N") {
       prop.N = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.N)>>(value);
       return;
    }
}

void TMEMSubSliceOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.N) attrs.append("N", prop.N);
}

::llvm::LogicalResult TMEMSubSliceOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "N", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TMEMSubSliceOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.N)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMSubSliceOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.N);
}

uint32_t TMEMSubSliceOp::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

void TMEMSubSliceOp::setN(uint32_t attrValue) {
  getProperties().N = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::IntegerAttr N) {
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().N = N;
  odsState.addTypes(result);
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::IntegerAttr N) {
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().N = N;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, uint32_t N) {
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().N = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), N);
  odsState.addTypes(result);
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, uint32_t N) {
  odsState.addOperands(src);
  odsState.getOrAddProperties<Properties>().N = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), N);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TMEMSubSliceOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TMEMSubSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TMEMSubSliceOp::verifyInvariantsImpl() {
  auto tblgen_N = getProperties().N; (void)tblgen_N;
  if (!tblgen_N) return emitOpError("requires attribute 'N'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_N, "N")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TMEMSubSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TMEMSubSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(srcRawType))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawType))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMEMSubSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getSrc().getType();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void TMEMSubSliceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMSubSliceOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TensorDescToTMAPtrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TensorDescToTMAPtrOpGenericAdaptorBase::TensorDescToTMAPtrOpGenericAdaptorBase(TensorDescToTMAPtrOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

} // namespace detail
TensorDescToTMAPtrOpAdaptor::TensorDescToTMAPtrOpAdaptor(TensorDescToTMAPtrOp op) : TensorDescToTMAPtrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TensorDescToTMAPtrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult TensorDescToTMAPtrOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TensorDescToTMAPtrOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TensorDescToTMAPtrOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TensorDescToTMAPtrOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TensorDescToTMAPtrOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TensorDescToTMAPtrOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TensorDescToTMAPtrOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TensorDescToTMAPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value desc) {
      auto ptrTy = triton::PointerType::get(odsBuilder.getI8Type(), 1);
      build(odsBuilder, odsState, ptrTy, desc);
    
}

void TensorDescToTMAPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type ptr, ::mlir::Value desc) {
  odsState.addOperands(desc);
  odsState.addTypes(ptr);
}

void TensorDescToTMAPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc) {
  odsState.addOperands(desc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorDescToTMAPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TensorDescToTMAPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult TensorDescToTMAPtrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TensorDescToTMAPtrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TensorDescToTMAPtrOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult TensorDescToTMAPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand descRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> descOperands(&descRawOperand, 1);  ::llvm::SMLoc descOperandsLoc;
  (void)descOperandsLoc;
  ::mlir::Type descRawType{};
  ::llvm::ArrayRef<::mlir::Type> descTypes(&descRawType, 1);
  ::mlir::Type ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> ptrTypes(&ptrRawType, 1);

  descOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(descRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(descRawType))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(ptrRawType))
    return ::mlir::failure();
  result.addTypes(ptrTypes);
  if (parser.resolveOperands(descOperands, descTypes, descOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorDescToTMAPtrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDesc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getDesc().getType();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
   _odsPrinter << getPtr().getType();
}

void TensorDescToTMAPtrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorDescToTMAPtrOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::WaitBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WaitBarrierOpGenericAdaptorBase::WaitBarrierOpGenericAdaptorBase(WaitBarrierOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WaitBarrierOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

} // namespace detail
WaitBarrierOpAdaptor::WaitBarrierOpAdaptor(WaitBarrierOp op) : WaitBarrierOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult WaitBarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WaitBarrierOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange WaitBarrierOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange WaitBarrierOp::getDepsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::llvm::LogicalResult WaitBarrierOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
    {

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) -> ::mlir::LogicalResult {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute WaitBarrierOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.operandSegmentSizes;
      auto attr = [&]() -> ::mlir::Attribute {
        return ::mlir::DenseI32ArrayAttr::get(ctx, propStorage);
      }();
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes", attr));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WaitBarrierOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    using ::llvm::hash_value;
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  using llvm::hash_value;
  return llvm::hash_combine(
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> WaitBarrierOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void WaitBarrierOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void WaitBarrierOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult WaitBarrierOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

::llvm::LogicalResult WaitBarrierOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void WaitBarrierOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value alloc, Value phase) {
    build(odsBuilder, odsState, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), /*deps=*/{});
    
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value alloc, Value phase, Value pred) {
    build(odsBuilder, odsState, alloc, phase, pred, /*deps=*/{});
    
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value alloc, Value phase, ValueRange deps) {
    build(odsBuilder, odsState, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), deps);
    
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::Value phase, /*optional*/::mlir::Value pred, ::mlir::ValueRange deps) {
  odsState.addOperands(alloc);
  odsState.addOperands(phase);
  if (pred)
    odsState.addOperands(pred);
  odsState.addOperands(deps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, (pred ? 1 : 0), static_cast<int32_t>(deps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::Value phase, /*optional*/::mlir::Value pred, ::mlir::ValueRange deps) {
  odsState.addOperands(alloc);
  odsState.addOperands(phase);
  if (pred)
    odsState.addOperands(pred);
  odsState.addOperands(deps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, (pred ? 1 : 0), static_cast<int32_t>(deps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WaitBarrierOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void WaitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult WaitBarrierOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult WaitBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WaitBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand phaseRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> phaseOperands(&phaseRawOperand, 1);  ::llvm::SMLoc phaseOperandsLoc;
  (void)phaseOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> predOperands;
  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depsOperands;
  ::llvm::SMLoc depsOperandsLoc;
  (void)depsOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);
  ::llvm::SmallVector<::mlir::Type, 1> depsTypes;

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  phaseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(phaseRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    predOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      predOperands.push_back(operand);
    }
  }
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("deps"))) {

  depsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(depsOperands))
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(allocRawType))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(depsTypes))
    return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, static_cast<int32_t>(predOperands.size()), static_cast<int32_t>(depsOperands.size())}), result.getOrAddProperties<WaitBarrierOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(phaseOperands, odsBuildableType0, phaseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(depsOperands, depsTypes, depsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WaitBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPhase();
  if (getPred()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getPred())
      _odsPrinter << value;
  }
  if (!getDeps().empty()) {
    _odsPrinter << ' ' << "deps";
    _odsPrinter << ' ';
    _odsPrinter << getDeps();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getAlloc().getType();
  if (!getDeps().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getDeps().getTypes();
  }
}

void WaitBarrierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::triton::gpu::SharedMemory::get());
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::WaitBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::WarpGroupDotOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WarpGroupDotOpGenericAdaptorBase::WarpGroupDotOpGenericAdaptorBase(WarpGroupDotOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WarpGroupDotOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::triton::InputPrecisionAttr WarpGroupDotOpGenericAdaptorBase::getInputPrecisionAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::triton::InputPrecisionAttr>(getProperties().inputPrecision);
  return attr;
}

::mlir::triton::InputPrecision WarpGroupDotOpGenericAdaptorBase::getInputPrecision() {
  auto attr = getInputPrecisionAttr();
  return attr.getValue();
}

::mlir::IntegerAttr WarpGroupDotOpGenericAdaptorBase::getMaxNumImpreciseAccAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().maxNumImpreciseAcc);
  return attr;
}

uint32_t WarpGroupDotOpGenericAdaptorBase::getMaxNumImpreciseAcc() {
  auto attr = getMaxNumImpreciseAccAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr WarpGroupDotOpGenericAdaptorBase::getIsAsyncAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().isAsync);
  return attr;
}

bool WarpGroupDotOpGenericAdaptorBase::getIsAsync() {
  auto attr = getIsAsyncAttr();
  return attr.getValue();
}

} // namespace detail
WarpGroupDotOpAdaptor::WarpGroupDotOpAdaptor(WarpGroupDotOp op) : WarpGroupDotOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult WarpGroupDotOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inputPrecision = getProperties().inputPrecision; (void)tblgen_inputPrecision;
  auto tblgen_isAsync = getProperties().isAsync; (void)tblgen_isAsync;
  auto tblgen_maxNumImpreciseAcc = getProperties().maxNumImpreciseAcc; (void)tblgen_maxNumImpreciseAcc;

  if (tblgen_inputPrecision && !((::llvm::isa<::mlir::triton::InputPrecisionAttr>(tblgen_inputPrecision))))
    return emitError(loc, "'ttng.warp_group_dot' op ""attribute 'inputPrecision' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2");

  if (tblgen_maxNumImpreciseAcc && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_maxNumImpreciseAcc))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_maxNumImpreciseAcc).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.warp_group_dot' op ""attribute 'maxNumImpreciseAcc' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_isAsync && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isAsync))))
    return emitError(loc, "'ttng.warp_group_dot' op ""attribute 'isAsync' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WarpGroupDotOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange WarpGroupDotOp::getUseCMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult WarpGroupDotOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.inputPrecision;
       auto attr = dict.get("inputPrecision");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `inputPrecision` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.isAsync;
       auto attr = dict.get("isAsync");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isAsync` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.maxNumImpreciseAcc;
       auto attr = dict.get("maxNumImpreciseAcc");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `maxNumImpreciseAcc` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute WarpGroupDotOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.inputPrecision;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inputPrecision",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.isAsync;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isAsync",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.maxNumImpreciseAcc;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("maxNumImpreciseAcc",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WarpGroupDotOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.inputPrecision.getAsOpaquePointer()), 
    llvm::hash_value(prop.isAsync.getAsOpaquePointer()), 
    llvm::hash_value(prop.maxNumImpreciseAcc.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> WarpGroupDotOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "inputPrecision")
      return prop.inputPrecision;

    if (name == "isAsync")
      return prop.isAsync;

    if (name == "maxNumImpreciseAcc")
      return prop.maxNumImpreciseAcc;
  return std::nullopt;
}

void WarpGroupDotOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inputPrecision") {
       prop.inputPrecision = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputPrecision)>>(value);
       return;
    }

    if (name == "isAsync") {
       prop.isAsync = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isAsync)>>(value);
       return;
    }

    if (name == "maxNumImpreciseAcc") {
       prop.maxNumImpreciseAcc = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.maxNumImpreciseAcc)>>(value);
       return;
    }
}

void WarpGroupDotOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.inputPrecision) attrs.append("inputPrecision", prop.inputPrecision);

    if (prop.isAsync) attrs.append("isAsync", prop.isAsync);

    if (prop.maxNumImpreciseAcc) attrs.append("maxNumImpreciseAcc", prop.maxNumImpreciseAcc);
}

::llvm::LogicalResult WarpGroupDotOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInputPrecisionAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps9(attr, "inputPrecision", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIsAsyncAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, "isAsync", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMaxNumImpreciseAccAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "maxNumImpreciseAcc", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult WarpGroupDotOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.inputPrecision)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.isAsync)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.maxNumImpreciseAcc)))
    return ::mlir::failure();
  return ::mlir::success();
}

void WarpGroupDotOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.inputPrecision);

  writer.writeOptionalAttribute(prop.isAsync);

  writer.writeOptionalAttribute(prop.maxNumImpreciseAcc);
}

::mlir::triton::InputPrecision WarpGroupDotOp::getInputPrecision() {
  auto attr = getInputPrecisionAttr();
  return attr.getValue();
}

uint32_t WarpGroupDotOp::getMaxNumImpreciseAcc() {
  auto attr = getMaxNumImpreciseAccAttr();
  return attr.getValue().getZExtValue();
}

bool WarpGroupDotOp::getIsAsync() {
  auto attr = getIsAsyncAttr();
  return attr.getValue();
}

void WarpGroupDotOp::setInputPrecision(::mlir::triton::InputPrecision attrValue) {
  getProperties().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WarpGroupDotOp::setMaxNumImpreciseAcc(uint32_t attrValue) {
  getProperties().maxNumImpreciseAcc = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WarpGroupDotOp::setIsAsync(bool attrValue) {
  getProperties().isAsync = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc, ::mlir::BoolAttr isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  if (inputPrecision) {
    odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  }
  if (maxNumImpreciseAcc) {
    odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;
  }
  if (isAsync) {
    odsState.getOrAddProperties<Properties>().isAsync = isAsync;
  }
  odsState.addTypes(d);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc, ::mlir::BoolAttr isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  if (inputPrecision) {
    odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  }
  if (maxNumImpreciseAcc) {
    odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;
  }
  if (isAsync) {
    odsState.getOrAddProperties<Properties>().isAsync = isAsync;
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WarpGroupDotOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc, ::mlir::BoolAttr isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  if (inputPrecision) {
    odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  }
  if (maxNumImpreciseAcc) {
    odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;
  }
  if (isAsync) {
    odsState.getOrAddProperties<Properties>().isAsync = isAsync;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc, bool isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);
  odsState.getOrAddProperties<Properties>().isAsync = odsBuilder.getBoolAttr(isAsync);
  odsState.addTypes(d);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc, bool isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);
  odsState.getOrAddProperties<Properties>().isAsync = odsBuilder.getBoolAttr(isAsync);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WarpGroupDotOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, /*optional*/::mlir::Value useC, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc, bool isAsync) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  if (useC)
    odsState.addOperands(useC);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);
  odsState.getOrAddProperties<Properties>().isAsync = odsBuilder.getBoolAttr(isAsync);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WarpGroupDotOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WarpGroupDotOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WarpGroupDotOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void WarpGroupDotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void WarpGroupDotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WarpGroupDotOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void WarpGroupDotOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.inputPrecision)
    properties.inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), ::mlir::triton::InputPrecision::IEEE);
  if (!properties.maxNumImpreciseAcc)
    properties.maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0);
  if (!properties.isAsync)
    properties.isAsync = odsBuilder.getBoolAttr(false);
}

::llvm::LogicalResult WarpGroupDotOp::verifyInvariantsImpl() {
  auto tblgen_inputPrecision = getProperties().inputPrecision; (void)tblgen_inputPrecision;
  auto tblgen_isAsync = getProperties().isAsync; (void)tblgen_isAsync;
  auto tblgen_maxNumImpreciseAcc = getProperties().maxNumImpreciseAcc; (void)tblgen_maxNumImpreciseAcc;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps9(*this, tblgen_inputPrecision, "inputPrecision")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_maxNumImpreciseAcc, "maxNumImpreciseAcc")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(*this, tblgen_isAsync, "isAsync")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType()))))
    return emitOpError("failed to verify that result's type matches accumulator's type");
  return ::mlir::success();
}

::llvm::LogicalResult WarpGroupDotOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WarpGroupDotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cOperands(&cRawOperand, 1);  ::llvm::SMLoc cOperandsLoc;
  (void)cOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> useCOperands;
  ::llvm::SMLoc useCOperandsLoc;
  (void)useCOperandsLoc;
  ::mlir::Type aRawType{};
  ::llvm::ArrayRef<::mlir::Type> aTypes(&aRawType, 1);
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type dRawType{};
  ::llvm::ArrayRef<::mlir::Type> dTypes(&dRawType, 1);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  cOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    useCOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      useCOperands.push_back(operand);
    }
  }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aRawType = type;
  }
  if (parser.parseStar())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dRawType = type;
  }
  for (::mlir::Type type : dTypes) {
    (void)type;
    if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((::llvm::isa<::mlir::Float8E4M3FNType>(elementType))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2Type>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(elementType))) || ((elementType.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(elementType))) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'d' must be ranked tensor of floating-point or integer values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  result.addTypes(dTypes);
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cOperands, dTypes[0], cOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(useCOperands, odsBuildableType0, useCOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WarpGroupDotOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getC();
  if (getUseC()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getUseC())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInputPrecisionAttr();
     if(attr && (attr == ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), ::mlir::triton::InputPrecision::IEEE)))
       elidedAttrs.push_back("inputPrecision");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getMaxNumImpreciseAccAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0)))
       elidedAttrs.push_back("maxNumImpreciseAcc");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsAsyncAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("isAsync");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getA().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "*";
  _odsPrinter << ' ';
  {
    auto type = getB().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getD().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::WarpGroupDotOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::WarpGroupDotWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WarpGroupDotWaitOpGenericAdaptorBase::WarpGroupDotWaitOpGenericAdaptorBase(WarpGroupDotWaitOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WarpGroupDotWaitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

uint32_t WarpGroupDotWaitOpGenericAdaptorBase::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
WarpGroupDotWaitOpAdaptor::WarpGroupDotWaitOpAdaptor(WarpGroupDotWaitOp op) : WarpGroupDotWaitOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult WarpGroupDotWaitOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitError(loc, "'ttng.warp_group_dot_wait' op ""requires attribute 'pendings'");

  if (tblgen_pendings && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pendings))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pendings).getType().isSignlessInteger(32)))))
    return emitError(loc, "'ttng.warp_group_dot_wait' op ""attribute 'pendings' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WarpGroupDotWaitOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange WarpGroupDotWaitOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WarpGroupDotWaitOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult WarpGroupDotWaitOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.pendings;
       auto attr = dict.get("pendings");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pendings` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute WarpGroupDotWaitOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.pendings;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pendings",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WarpGroupDotWaitOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.pendings.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> WarpGroupDotWaitOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "pendings")
      return prop.pendings;
  return std::nullopt;
}

void WarpGroupDotWaitOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "pendings") {
       prop.pendings = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pendings)>>(value);
       return;
    }
}

void WarpGroupDotWaitOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.pendings) attrs.append("pendings", prop.pendings);
}

::llvm::LogicalResult WarpGroupDotWaitOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getPendingsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "pendings", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult WarpGroupDotWaitOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.pendings)))
    return ::mlir::failure();
  return ::mlir::success();
}

void WarpGroupDotWaitOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.pendings);
}

uint32_t WarpGroupDotWaitOp::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

void WarpGroupDotWaitOp::setPendings(uint32_t attrValue) {
  getProperties().pendings = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  odsState.addTypes(outputs);
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = pendings;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WarpGroupDotWaitOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, uint32_t pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  odsState.addTypes(outputs);
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, uint32_t pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WarpGroupDotWaitOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WarpGroupDotWaitOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WarpGroupDotWaitOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WarpGroupDotWaitOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

void WarpGroupDotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WarpGroupDotWaitOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult WarpGroupDotWaitOp::verifyInvariantsImpl() {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitOpError("requires attribute 'pendings'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_pendings, "pendings")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(0).getType()) == (this->getODSResults(0).getType()) && (this->getODSResults(0).getType()) == (this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that all of {inputs, outputs} have same type");
  return ::mlir::success();
}

::llvm::LogicalResult WarpGroupDotWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult WarpGroupDotWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed(WarpGroupDotWaitOp::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.getRawProperties(),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
  return ::mlir::success();
}

void WarpGroupDotWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputs().getTypes();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::WarpGroupDotWaitOp)


#endif  // GET_OP_CLASSES

