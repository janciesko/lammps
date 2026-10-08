# preset that enables KOKKOS against an external kokkos-stdexec installation.
# Set KokkosStdexec_ROOT to the kokkos-stdexec install prefix or source tree,
# Kokkos_ROOT to the Kokkos used to build it, and stdexec_ROOT to stdexec.
set(PKG_KOKKOS ON CACHE BOOL "" FORCE)
set(EXTERNAL_KOKKOS_STDEXEC ON CACHE BOOL "" FORCE)

# hide deprecation warnings temporarily for stable release
set(Kokkos_ENABLE_DEPRECATION_WARNINGS OFF CACHE BOOL "" FORCE)
