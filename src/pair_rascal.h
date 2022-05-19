/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(rascal,PairRASCAL);
// clang-format on
#else

#ifndef LMP_PAIR_RASCAL_H
#define LMP_PAIR_RASCAL_H

#include "pair.h"

#include "rascal/representations/calculator_spherical_invariants.hh"
#include "rascal/models/sparse_kernels.hh"
#include "rascal/models/sparse_points.hh"
#include "rascal/math/utils.hh"

#include "rascal/structure_managers/structure_manager_lammps.hh"
#include "rascal/structure_managers/adaptor_center_contribution.hh"
#include "rascal/structure_managers/adaptor_strict.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/structure_manager_collection.hh"
// COMMENT(alex) I think we don't need this if we import h files
//extern "C" {
//int quip_lammps_api_version();
//void quip_lammps_wrapper(int *, int *, int *, int *, int *, int *, int *, int *, int *, double *,
//                         int *, int *, double *, double *, double *, double *, double *, double *);
//void quip_lammps_potential_initialise(int *, int *, double *, char *, int *, char *, int *);
//}

namespace LAMMPS_NS {

// 0 none (no addiditional log to lammps)
// 1 debug (force and energy calculation of rascal)
// 2 trace (neighbourlist of rascal)
enum class RASCAL_LOG { NONE = 0, DEBUG = 1, TRACE = 2 };

class PairRASCAL : public Pair {
 public:
  PairRASCAL(class LAMMPS *);
  ~PairRASCAL();

  void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void allocate();

 private:
  double cutoff;
  int *map;           // mapping from atom types to elements
  char *rascal_file;    // mapping from atom types to elements
  RASCAL_LOG log_level;
  std::shared_ptr<rascal::CalculatorSphericalInvariants> calculator{};
  std::shared_ptr<rascal::SparseKernel> kernel{};
  rascal::SparsePointsBlockSparse<rascal::CalculatorSphericalInvariants> sparse_points{};
  //std::vector<double> weights_vec{};
  rascal::math::Vector_t weights{};
  std::map<std::string, double> self_contributions{};
  std::vector<int> rascal_atom_types{}; 
  //std::shared_ptr<rascal::StructureManagerLammps> root_manager{rascal::make_structure_manager<rascal::StructureManagerLammps>()};
  //std::shared_ptr<rascal::StructureManager<rascal::StructureManagerLammps<rascal::AdaptorCenterContribution<rascal::AdaptorStrict>>>>{};
  rascal::ManagerCollection<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict> managers{};

};

}    // namespace LAMMPS_NS

#endif
#endif
