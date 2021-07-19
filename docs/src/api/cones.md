# Cones module

See the [Cone interface and predefined cones](@ref) section.

```@meta
CurrentModule = Hypatia.Cones
```

```@docs
Cones
Cone
```

## Array utilities

```@docs
vec_length
vec_copyto!
svec_length
svec_side
svec_idx
block_idxs
scale_svec!
smat_to_svec!
svec_to_smat!
```

## Cone oracles

```@docs
dimension
get_nu
set_initial_point!
is_feas
is_dual_feas
grad
hess
inv_hess
hess_prod!
inv_hess_prod!
use_dder3
dder3
```

## Predefined cone types

```@docs
Nonnegative
PosSemidefTri
DoublyNonnegativeTri
PosSemidefTriSparse
LinMatrixIneq
EpiNormInf
EpiNormEucl
EpiPerSquare
EpiNormSpectral
MatrixEpiPerSquare
GeneralizedPower
HypoPowerMean
HypoGeoMean
HypoRootdetTri
HypoPerLog
HypoPerLogdetTri
EpiPerSepSpectral
EpiRelEntropy
EpiTrRelEntropyTri
WSOSInterpNonnegative
WSOSInterpPosSemidefTri
WSOSInterpEpiNormEucl
WSOSInterpEpiNormOne
```

### Helpers for PosSemidefTriSparse

```@docs
PSDSparseImpl
PSDSparseDense
PSDSparseCholmod
```

### EpiPerSepSpectral helpers

```@docs
ConeOfSquares
VectorCSqr
MatrixCSqr
vector_dim
SepSpectralFun
NegLogSSF
NegEntropySSF
NegSqrtSSF
NegPower01SSF
Power12SSF
```
