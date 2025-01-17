# preset that turns on all existing packages off. can be used to reset
# an existing package selection without losing any other settings

set(ALL_PACKAGES ASPHERE BODY CLASS2 COLLOID COMPRESS CORESHELL DIPOLE GPU
  GRANULAR KIM KOKKOS KSPACE LATTE MANYBODY MC MESSAGE MISC MLIAP MOLECULE
  MPIIO MSCG OPT PERI PLUGIN POEMS PYTHON QEQ REPLICA RIGID SHOCK SNAP SPIN
  SRD VORONOI
  USER-ADIOS USER-ATC USER-AWPMD USER-BROWNIAN USER-BOCS USER-CGDNA USER-CGSDK
  USER-COLVARS USER-DIFFRACTION USER-DPD USER-DRUDE USER-EFF USER-FEP USER-H5MD
  USER-HDNNP USER-INTEL USER-LB USER-MANIFOLD USER-MDI USER-MEAMC USER-MESODPD
  USER-MESONT USER-MGPT USER-MISC USER-MOFFF USER-MOLFILE USER-NETCDF USER-OMP
  USER-PACE USER-PHONON USER-PLUMED USER-PTM USER-QMMM USER-QTB USER-QUIP USER-RANN
  USER-REACTION USER-REAXC USER-SCAFACOS USER-SDPD USER-SMD USER-SMTBQ USER-SPH
  USER-TALLY USER-UEF USER-VTK USER-YAFF USER-DIELECTRIC)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} OFF CACHE BOOL "" FORCE)
endforeach()
