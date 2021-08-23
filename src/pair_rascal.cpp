// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Alexander Goscinski (alexander.goscinski@epfl.ch)
------------------------------------------------------------------------- */

#include "pair_rascal.h"
#include <sched.h>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "lattice.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

#include "rascal/models/sparse_kernel_predict.hh"

#include "rascal/utils/json_io.hh"


using namespace LAMMPS_NS;

// copy paste pair_kim.cpp might be later useful
static constexpr const char *const cite_openkim =
  "OpenKIM: https://doi.org/10.1007/s11837-011-0102-6\n\n"
  "@Article{tadmor:elliott:2011,\n"
  " author = {E. B. Tadmor and R. S. Elliott and J. P. Sethna and R. E. Miller "
  "and C. A. Becker},\n"
  " title = {The potential of atomistic simulations and the {K}nowledgebase of "
  "{I}nteratomic {M}odels},\n"
  " journal = {{JOM}},\n"
  " year =    2011,\n"
  " volume =  63,\n"
  " number =  17,\n"
  " pages =   {17},\n"
  " doi =     {10.1007/s11837-011-0102-6}\n"
  "}\n\n";



//using Representation_t = CalculatorSphericalExpansion;

//using Manager_t = AdaptorStrict<
//    AdaptorCenterContribution<AdaptorNeighbourList<StructureManagerCenters>>>;
//using Prop_t = typename CalculatorSphericalExpansion::Property_t<Manager_t>;
//using PropGrad_t =
//    typename CalculatorSphericalExpansion::PropertyGradient_t<Manager_t>;

/* ---------------------------------------------------------------------- */

PairRASCAL::PairRASCAL(LAMMPS *lmp) : Pair(lmp)
{
  // COMMENT(alex) copy from quip, havent understood this yet
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  no_virial_fdotr_compute = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  map = nullptr;
  rascal_file = nullptr;
}

PairRASCAL::~PairRASCAL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
  delete [] rascal_file;
}

void PairRASCAL::compute(int eflag, int vflag)
{

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::compute start" << std::endl;
  int inum, jnum, sum_num_neigh, ii, jj, i;
  int *ilist;
  int *jlist;
  int *numneigh, **firstneigh;
  double *lattice;
  int *pbc;

  // COMMENT(alex) as far as I have observed nlocal == inum, so I could remove this and only use inum
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *type = atom->type;
  // TODO(alex) tagint should be properly handeled
  //            tagint can be also int or int64_t depending on compile flags see LAMMPS_BIGBIG
  int *tag = atom->tag;

  double **x = atom->x;
  double **f = atom->f;

  int const tot_num = atom->nlocal + atom->nghost;

  // seems to be lammps intern
  ev_init(eflag,vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " <<  "Lammps number of neighbours: nlocal + nghost = " << nlocal << "+" << nghost <<std::endl;
    std::cout << sched_getcpu() << ": " <<  "Lammps inum = " << inum << std::endl;
  }
  
  // in domain decomposition, domains can have 0 center atoms
  if (inum == 0) {
    if (this->log_level >= RASCAL_LOG::DEBUG)
      std::cout << sched_getcpu() << ": " << "inum is 0 set forces to zero" << std::endl;
    for (ii = 0; ii < inum; ii++) {
       for (jj = 0; jj < 3; jj++) {
          f[ii][jj] = 0;
       }
    }

    if (this->log_level >= RASCAL_LOG::DEBUG)
      std::cout << sched_getcpu() << ": " << "inum is 0 set structure energy to zero" << std::endl;
    if (eflag_global) {
      eng_vdwl = 0;
    }
    
    // TODO(alex) to obtain per atom properties it requires changes in the prediction interface of rascal
    //if (eflag_atom) {
    //  for (ii = 0; ii < ntotal; ii++) {
    //    eatom[ii] = rascal_energies(0,0)/ntotal;
    //  }
    //}

    if (this->log_level >= RASCAL_LOG::DEBUG)
      std::cout << sched_getcpu() << ": " << "inum is 0 set structure stress to zero" << std::endl;
    if (vflag_global) {
        virial[0] = 0;
        virial[1] = 0;
        virial[2] = 0;
        virial[3] = 0;
        virial[4] = 0;
        virial[5] = 0;
    }
    return;
  }

  lattice = new double [9];
  lattice[0] = domain->xprd;
  lattice[1] = 0.0;
  lattice[2] = 0.0;
  lattice[3] = domain->xy;
  lattice[4] = domain->yprd;
  lattice[5] = 0.0;
  lattice[6] = domain->xz;
  lattice[7] = domain->yz;
  lattice[8] = domain->zprd;

  // TODO(alex) can be set more globally at the initialization process
  pbc = new int [3];
  pbc[0] = domain->xperiodic;
  pbc[1] = domain->yperiodic;
  pbc[2] = domain->zperiodic;


  if (this->log_level >= RASCAL_LOG::DEBUG) {
    // COMMENT(alex) in my cases corners contains only zeros, can be probably removed
    //std::cout << sched_getcpu() << ": " << "domain box corners" << std::endl;
    //for (int i=0; i < 8 ; i++) {
    //  std::cout << sched_getcpu() << ": ";
    //  for (int p=0; p < 3 ; p++) {
    //    std::cout << domain->corners[i][p] << " ";
    //  }
    //  std::cout << std::endl;
    //}
    for (int p=0; p < 3 ; p++) {
    std::cout << sched_getcpu() << ": "
              << "sublo["<< p << "] " << "subhi["<< p << "] " << domain->sublo[p] << " " << domain->subhi[p] << std::endl;
    }
  }

  if (this->log_level >= RASCAL_LOG::TRACE) {
    std::cout << sched_getcpu() << ": " << "Lammps lattice: "
              << domain->lattice->a1[0] << ", " << domain->lattice->a1[1] << ", " << domain->lattice->a1[2] << "; "
              << domain->lattice->a2[0] << ", " << domain->lattice->a2[1] << ", " << domain->lattice->a2[2] << "; "
              << domain->lattice->a3[0] << ", " << domain->lattice->a3[1] << ", " << domain->lattice->a3[2] 
              << std::endl;

    std::cout << sched_getcpu() << ": " << "Lammps (domain) cell: "
              << domain->xprd << ", " << 0 << ", " << 0 << "; "
              << domain->xy << ", " << domain->yprd << ", " << 0 << "; "
              << domain->xz << ", " << domain->yz << ", " << domain->zprd 
              << std::endl;
    std::cout << sched_getcpu() << ": " << "Lammps pbc: "
              << domain->xperiodic << ", " << domain->yperiodic << ", " << domain->zperiodic
              << std::endl;
    std::cout << sched_getcpu() << ": " << "Lammps position list:\n";
    for (int i=0; i < tot_num ; i++) {
      std::cout << sched_getcpu() << ": " << "center " << i << ", lammps atom tag " << atom->tag[i] << ", ";
      std::cout << "position ";
      for (int j=0; j < 3; j++) {
        std::cout << atom->x[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << sched_getcpu() << ": " << "Lammps neighbor list:\n";
    for (int i=0; i < inum; i++) {
      std::cout << sched_getcpu() << ": " << "center " << list->ilist[i] << ", ";
      std::cout << "position ";
      for (int p=0; p < 3; p++) {
        std::cout << atom->x[p][p] << " ";
      }
      std::cout << std::endl;
      for (int j=0; j < numneigh[i]; j++) {
        std::cout << sched_getcpu() << ": " << "  neigh " << list->firstneigh[i][j] << ", position ";
        for (int p=0; p < 3; p++) {
          std::cout << atom->x[list->firstneigh[i][j]][p] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::shared_ptr<rascal::AdaptorStrict<rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>> manager = *managers.begin();
  manager->update(inum, tot_num, ilist, numneigh, firstneigh, x, f, type, eatom, vatom, rascal_atom_types, tag, lattice, pbc);

  if (this->log_level >= RASCAL_LOG::TRACE) {
      // rascal manager/adaptors do not have get_cell method implemented,
      // therefore we have to get the information in this "hacky" way
      // TODO getcpu for lattice
      std::cout << sched_getcpu() << ": " << "rascal lattice:" << rascal::extract_underlying_manager<0>(manager)->get_cell() << std::endl;
      std::cout << sched_getcpu() << ": " << "rascal pbc: " << rascal::extract_underlying_manager<0>(manager)->get_periodic_boundary_conditions().transpose() << std::endl;
      std::cout << sched_getcpu() << ": " << "rascal cell volume: " << rascal::extract_underlying_manager<0>(manager)->get_cell_volume() << std::endl;
      std::cout << sched_getcpu() << ": " << "rascal manager->offsets\n";
      for (int k=0; k < manager->offsets.size(); k++) {
         std::cout << sched_getcpu() << ": ";
         for (int p=0; p < manager->offsets[k].size(); p++) {
           std::cout << manager->offsets[k][p] << ", ";
         }
         std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << sched_getcpu() << ": " << "rascal manager->nb_neigh\n";
      for (int k=0; k < manager->nb_neigh.size(); k++) {
         std::cout << sched_getcpu() << ": ";
         for (int p=0; p < manager->nb_neigh[k].size(); p++) {
           std::cout << manager->nb_neigh[k][p] << ", ";
         }
         std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << sched_getcpu() << ": " << "rascal manager->atom_tag_list\n";
      for (int k=0; k < manager->atom_tag_list.size(); k++) {
         std::cout << sched_getcpu() << ": Order " << k << ": ";
         for (int p=0; p < manager->atom_tag_list[k].size(); p++) {
           std::cout << manager->atom_tag_list[k][p] << ", ";
         }
         std::cout << std::endl;
      }
      std::cout << std::endl;
      
      std::cout << sched_getcpu() << ": " << "rascal manager->neighbours_cluster_index: ";
      for (int k=0; k < manager->neighbours_cluster_index.size(); k++) {
         std::cout << manager->neighbours_cluster_index[k] << ", ";
      }
      std::cout << std::endl;

      std::cout << sched_getcpu() << ": " << "rascal neighbor list without ghosts\n";
      for (auto atom : manager) {
        std::cout << sched_getcpu() << ": " << "center atom tag " << atom.get_atom_tag() << ", "
                  << "cluster index " << atom.get_cluster_index()
                  << std::endl;
        for (auto pair : atom.pairs_with_self_pair()) {
          std::cout << sched_getcpu() << ": " << "  pair (" << atom.get_atom_tag() << ", "
                    << pair.get_atom_tag() << "): " 
                    << "global index " << pair.get_global_index() << ", "
                    << "pair dist " << manager->get_distance(pair)  << ", " 
                    << "direction vector " << manager->get_direction_vector(pair).transpose()
                    << std::endl;
        }
      }
      std::cout << std::endl;

      std::cout << sched_getcpu() << ": " << "rascal neighbor list with ghosts\n";
      if (inum > 0) {
        for (auto atom : manager->with_ghosts()) {
          std::cout << sched_getcpu() << ": " << "center atom tag " << atom.get_atom_tag() << ", "
                    << "cluster index " << atom.get_cluster_index()
                    << std::endl;
          for (auto pair : atom.pairs_with_self_pair()) {
            std::cout << sched_getcpu() << ": " << "  pair (" << atom.get_atom_tag() << ", "
                      << pair.get_atom_tag() << "): "
                      << "global index " << pair.get_global_index()
                      << std::endl;
          }
        }
      } else {
        std::cout << sched_getcpu() << ": " << "inum is 0 therefore we do not print ghost neighbours, because rascal cannot handle it" << std::endl;
      }
      std::cout << std::endl;
  }

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Computing representation..." << std::endl;
  calculator->compute(managers);
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Computed representation." << std::endl;
 
  // representation vector can be printed if needed but even for trace
  // information it is too large, besides this it usally does not contain
  // information helpful for debugging which you cannot derive from other outputs
  //auto && expansions_coefficients{*manager->template get_property<rascal::CalculatorSphericalInvariants::template Property_t<rascal::AdaptorStrict<
  // rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>>>(
  //  calculator->get_name())};
  //if (this->log_level >= RASCAL_LOG::TRACE) {
  //  for (auto atom : manager) {
  //    std::cout << sched_getcpu() << ": " << expansions_coefficients[atom].get_full_vector().transpose() << std::endl;
  //  }
  //}


  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Compute KNM" << std::endl;
  rascal::math::Matrix_t KNM{kernel->compute(*calculator, managers, sparse_points)};

  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "KNM shape " << KNM.rows() << ", " << KNM.cols() << std::endl;
    std::cout << sched_getcpu() << ": " << "weights shape " << weights.rows() << ", " << weights.cols() << std::endl;
  }

  // predict energies 
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Compute energies" << std::endl;
  rascal::math::Matrix_t rascal_energies = KNM * weights.transpose();
  for (auto center : manager) {
    rascal_energies(0,0) += self_contributions[std::to_string(center.get_atom_type())];
  }
  //baseline  = ..
  //energies += baseline;
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "rascal energies with shape (" << rascal_energies.rows() << ", " << rascal_energies.cols() << "):\n"
              << sched_getcpu() << ": " <<  rascal_energies
              << std::endl;
  }

  // predict forces
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Compute forces" << std::endl;
  std::string force_name = rascal::compute_sparse_kernel_gradients(
          *calculator, *kernel, managers, sparse_points, weights);
  auto && gradients{*manager->template get_property<
      rascal::Property<double, 1, rascal::AdaptorStrict<
      rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>, 1, 3>>(force_name, true)};
  // REMINDER(alex) with the line below the matrix should be copied 
  //rascal::math::Matrix_t rascal_force = Eigen::Map<const rascal::math::Matrix_t>(
  //     gradients.view().data(), manager->size(), 3);
  auto rascal_forces = Eigen::Map<const rascal::math::Matrix_t>(
       gradients.view().data(), manager->size(), 3);
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "rascal forces with shape (" << rascal_forces.rows() << ", " << rascal_forces.cols() << "):\n";
    for (int i=0; i < rascal_forces.rows(); i++) {
      std::cout << sched_getcpu() << ": " << rascal_forces.row(i) << std::endl;
    }
  }

  // predict stress
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Compute stress" << std::endl;
  std::string neg_stress_name = rascal::compute_sparse_kernel_neg_stress(
      *calculator, *kernel, managers, sparse_points, weights);
  auto && gradients_neg_stress {
      *manager->template get_property<rascal::Property<double, 0, rascal::AdaptorStrict<
      rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>, 6>>(
          neg_stress_name, true)};
  // TODO(alex) in rascal cpp the stress is divided by volume, but lammps does this later too
  //            so we undo the division from rascal, this should be properly solved with a
  //            flag in rascal cpp or by removing it on the cpp side and adding it to the python side

  Eigen::Map<const rascal::math::Vector_t> rascal_negative_stress =
     Eigen::Map<const rascal::math::Vector_t>(gradients_neg_stress.view().data(), 6);
  //rascal_negative_stress(0) *= rascal::extract_underlying_manager<0>(manager)->get_cell_volume();

  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "rascal negative_stress with shape (" << rascal_negative_stress.rows() << ", " << rascal_negative_stress.cols() << "):\n"
              << sched_getcpu() << ": " << rascal_negative_stress
              << std::endl;
  }

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Copy forces to lammps" << std::endl;
  for (ii = 0; ii < inum; ii++) {
     for (jj = 0; jj < 3; jj++) {
        f[ii][jj] = -rascal_forces(ii, jj);
     }
  }

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Copy structure energy to lammps" << std::endl;
  if (eflag_global) {
    eng_vdwl = rascal_energies(0,0);
  }
  
  // TODO(alex) to obtain per atom properties it requires changes in the prediction interface of rascal
  if (eflag_atom) {
    for (ii = 0; ii < ntotal; ii++) {
      eatom[ii] = rascal_energies(0,0)/ntotal;
    }
  }

  // rascal: xx, yy, zz, yz, xz, xy
  // lammps: xx, yy, zz, xy, xz, yz,
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "Copy structure stress to lammps" << std::endl;
  if (vflag_global) {
      virial[0] = rascal_negative_stress(0) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
      virial[1] = rascal_negative_stress(1) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
      virial[2] = rascal_negative_stress(2) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
      virial[3] = rascal_negative_stress(5) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
      virial[4] = rascal_negative_stress(4) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
      virial[5] = rascal_negative_stress(3) * rascal::extract_underlying_manager<0>(manager)->get_cell_volume();
  }

  // TODO(alex) to obtain per atom properties it requires changes in the prediction interface of rascal
  //if (vflag_atom) {
  //  int iatom = 0;
  //   for (ii = 0; ii < ntotal; ii++) {
  //     vatom[ii][0] += 0.5; 
  //     vatom[ii][1] += 0.5; 
  //     vatom[ii][2] += 0.5; 
  //     vatom[ii][3] += 0.5; 
  //     vatom[ii][4] += 0.5; 
  //     vatom[ii][5] += 0.5; 
  //     iatom += 9;
  //   }
  //}
  
  delete [] lattice;
  delete [] pbc;
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::compute end" << std::endl;
}

// I think rascal also require in metal units, since QUIP does
//  QUIP potentials are parameterized in electron-volts and Angstroms and therefore should be used with LAMMPS metal units.
// https://docs.lammps.org/pair_quip.html

void PairRASCAL::settings(int narg, char ** /* arg */)
{
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "PairRASCAL::settings start" << std::endl;
  }
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");

  if (strcmp("metal",update->unit_style) != 0)
    error->all(FLERR,"Rascal potentials require 'metal' units");
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "PairRASCAL::settings end" << std::endl;
  }
}

void PairRASCAL::allocate()
{
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "PairRASCAL::allocate start" << std::endl;
  }
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag,n+1,n+1,"pair:setflag");
  cutsq = memory->create(cutsq,n+1,n+1,"pair:cutsq");
  map = new int[n+1];
  if (this->log_level >= RASCAL_LOG::DEBUG) {
    std::cout << sched_getcpu() << ": " << "PairRASCAL::allocate end" << std::endl;
  }
}

// For initialization, gives the input of of pair_coeff
// e.g. If your file 
// pair_style      quip
// pair_coeff      * * gap_example.xml "Potential xml_label=GAP_2014_5_8_60_17_10_38_466" 14
// arg contains the arguments ['*', '*', 'gap_example', 'Potential xml_label=GAP_2014_5_8_60_17_10_38_466', '14']

void PairRASCAL::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int n = atom->ntypes;
  if (narg != (4 + n))
    error->all(FLERR,"Number of arguments {} is not correct, "
                                 "it should be {}", narg, 4 + n);

  // ensure I,J args are * *
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (strcmp(arg[2],"none") == 0) {
    log_level = RASCAL_LOG::NONE; 
  } else if (strcmp(arg[2],"debug") == 0) {
    log_level = RASCAL_LOG::DEBUG; 
  } else if (strcmp(arg[2],"trace") == 0) {
    log_level = RASCAL_LOG::TRACE; 
  } else {
    error->all(FLERR,"Rascal log level only supports {info|debug|trace}");
  }

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::coeff start" << std::endl;

  rascal_file = utils::strdup(arg[3]);
  rascal_atom_types.reserve(n);
  for (int i=0; i<n; i++) {
    rascal_atom_types.emplace_back(std::atoi(arg[i+4]));
    if (this->log_level >= RASCAL_LOG::DEBUG)
      std::cout << sched_getcpu() << ": " << "rascal_atom_types " << rascal_atom_types[i] << std::endl;
  }

  //// begin within rascal (code would be later moved into rascal)
  // here we load model file 
  json input;
  try {
    input = rascal::json_io::load(rascal_file);
  } catch (const std::exception& e) {
    std::stringstream error{};
    error << "Error: "
          << "Loading json failed. "
          << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
          << e.what();
    throw std::runtime_error(error.str());
  }

  json init_params;
  json X_train;
  try {
    init_params = input.at("init_params").template get<json>();
    X_train = init_params.at("X_train").template get<json>();
  } catch (const std::exception& e) {
    std::stringstream error{};
    error << "Error: "
          << "Loading init parameters from json failed. "
          << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
          << e.what();
    throw std::runtime_error(error.str());
  }

  // sparse points
  try {
    json sparse_data = X_train.at("data").template get<json>();
    json sparse_input = sparse_data.at("sparse_points").template get<json>();
    //sparse_points = rascal::SparsePointsBlockSparse<rascal::CalculatorSphericalInvariants>();
    rascal::from_json(sparse_input, sparse_points);
  } catch (const std::exception& e) {
    std::stringstream error{};
    error << "Error: "
          << "Loading sparse points from json failed. "
          << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
          << e.what();
    throw std::runtime_error(error.str());
  }

  // kernel
  json kernel_params;
  try {
    kernel_params = init_params.at("kernel").template get<json>();
    json kernel_data = kernel_params.at("data").template get<json>();
    json kernel_cpp_params = kernel_data.at("cpp_kernel").template get<json>();
    kernel = std::make_shared<rascal::SparseKernel>(kernel_cpp_params);
  } catch (const std::exception& e) {
    std::stringstream error{};
    error << "Error\n"
          << "Loading kernel from json failed. "
          << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
          << e.what()
          << std::endl;
    throw std::runtime_error(error.str());
  }

  // calculator
  json representation_cpp_params;
  try {
    json kernel_init_params = kernel_params.at("init_params").template get<json>();
    json kernel_representation = kernel_init_params.at("representation").template get<json>();
    json kernel_representation_data = kernel_representation.at("data").template get<json>();
    representation_cpp_params = kernel_representation_data.at("cpp_representation").template get<json>();
    calculator = std::make_shared<rascal::CalculatorSphericalInvariants>(representation_cpp_params);
  } catch (const std::exception& e) {
    std::stringstream error{};
    error << "Error\n"
          << "Loading calculator from json failed. "
          << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
          << e.what()
          << std::endl;
    throw std::runtime_error(error.str());
  }
  self_contributions = init_params.at("self_contributions").template get<std::map<std::string, double>>();

  // weights
  // COMMENT(alex) how weights could be loaded from a std::vector<double>, in this case Eigen provides simplifying utilities
  //std::vector<double> weights_vec = init_params.at("weights").template get<json>().at(1).template get<std::vector<double>>();
  //weights = Eigen::Map<rascal::math::Vector_t>(weights_vec.data(), static_cast<long int>(weights_vec.size()));
  std::vector<double> weights_vec;
  try {
    weights_vec = init_params.at("weights").template get<json>().at(1).template get<std::vector<double>>();
  } catch (const std::exception& e) {
    std::cerr << "Error\n"
              << "Loading weights from json failed. "
              << "In file " << __FILE__ << " (line " << __LINE__ << ")\n"
              << e.what()
              << std::endl;
  }
  if (sparse_points.size() != weights_vec.size()) {
    std::cerr << "weight size and sparse_points size disagree "
              << "In file " << __FILE__ << " (line " << __LINE__ << ")"
              << std::endl;
  }
  weights = rascal::math::Vector_t(weights_vec.size());
  for (unsigned int i=0; i < weights_vec.size(); i++) {
    weights(i) = weights_vec[i];
  }

  // cutoff
  cutoff = representation_cpp_params.at("cutoff_function").template get<json>().at("cutoff").template get<json>().at("value").template get<double>();


  // COMMENT(alex) copied from QUIP, havent understood this
  // clear setflag since coeff() called once with I,J = * *
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

  auto root_manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();

  json adaptors_input = {
      {
        {"initialization_arguments", {}},
        {"name", "centercontribution"},
      },
      {
        {"initialization_arguments", {{"cutoff", cutoff}}},
        {"name", "strict"}
      }
  };
  auto manager = rascal::stack_adaptors<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(root_manager, adaptors_input);
  managers = rascal::ManagerCollection<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(adaptors_input);

  managers.add_structure(manager);

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::coeff end" << std::endl;
  //// end within rascal 
}

void PairRASCAL::init_style()
{
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::init_style start" << std::endl;
  // COMMENT(alex) Copy from quip, I am not sure yet how this work.
  //               My guess is that it is not necessary putting does
  //               not require extra MPI communication so they enforce it to set off 
  if (force->newton_pair != 1)
    error->all(FLERR,"Pair style rascal requires newton pair on");

  // Initialise neighbor list
  int irequest_full = neighbor->request(this);
  // COMMENT(alex) could be generalized to support both types but I don't know if people need this
  if (neighbor->requests[irequest_full]->full) {
    std::cout << sched_getcpu() << ": " << "WARNING: Found request for half neighborlist, but rascal pair "
                 "potential only works with full neighborlist. Setting "
                 "it to full neighborlist."
              << std::endl;
  }
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;

  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::init_style end" << std::endl;
}

double PairRASCAL::init_one(int /*i*/, int /*j*/)
{
  if (this->log_level >= RASCAL_LOG::DEBUG)
    std::cout << sched_getcpu() << ": " << "PairRASCAL::init_one return with cutoff " << cutoff << std::endl;
  return cutoff;
}
