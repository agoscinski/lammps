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
   Contributing authors: Albert Bartok (Cambridge University)
                         Aidan Thompson (Sandia, athomps@sandia.gov)
------------------------------------------------------------------------- */

#include "pair_rascal.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

#include "rascal/representations/calculator_spherical_invariants.hh"
#include "rascal/structure_managers/structure_manager_lammps.hh"
#include "rascal/structure_managers/structure_manager_centers.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/adaptor_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_strict.hh"



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
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  no_virial_fdotr_compute = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  map = nullptr;
  quip_potential = nullptr;
  rascal_file = nullptr;
}

PairRASCAL::~PairRASCAL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
  delete [] quip_potential;
  delete [] rascal_file;
}

void PairRASCAL::compute(int eflag, int vflag)
{
  std::cout << "PairRASCAL::compute start" << std::endl;
  int inum, jnum, sum_num_neigh, ii, jj, i, iquip;
  int *ilist;
  int *jlist;
  int *numneigh, **firstneigh;
  int *quip_num_neigh, *quip_neigh, *atomic_numbers;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *type = atom->type;
  tagint *tag = atom->tag;

  double **x = atom->x;
  double **f = atom->f;

  int const tot_num = atom->nlocal + atom->nghost;

  // seems to be lammps intern
  ev_init(eflag,vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  //json hypers;
  //rascal::CalculatorSphericalInvariants calculator{hypers};

  // would be loaded in the coeff step like this, adaptors
  //double cutoff{5};
  //json adaptors;
  //json ad1{{"name", "AdaptorNeighbourList"},
  //         {"initialization_arguments", {{"cutoff", cutoff}}}};
  //json ad1b{{"name", "AdaptorCenterContribution"},
  //          {"initialization_arguments", {}}};
  //json ad2{{"name", "AdaptorStrict"},
  //         {"initialization_arguments", {{"cutoff", cutoff}}}};
  //adaptors.emplace_back(ad1);
  //adaptors.emplace_back(ad1b);
  //adaptors.emplace_back(ad2);

  //   int inum, int tot_num, int * ilist,
  //   int * numneigh, int ** firstneigh,
  //   double ** x, double ** f, int * type,
  //   double * eatom, double ** vatom
  std::cout << "Creating root manager..." << std::endl;
  auto manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();
  std::cout << "Created root manager." << std::endl;
  // & ? needed somewhere
  std::cout << "Update root manager..." << std::endl;
  manager->update(inum, tot_num, ilist, numneigh, firstneigh, x, f, type, eatom, vatom);
  //manager->update(inum, tot_num, ilist, numneigh,
  //               static_cast<int **>(firstneigh), double  **(x), double **(f), type,
  //               eatom, static_cast<double **>(vatom));

  std::cout << "Updated root manager." << std::endl;
  //auto man_cen = rascal::make_structure_manager<rascal::StructureManagerCenters>();
  // is strict required? rascal::AdaptorStrict
  //auto man = rascal::stack_adaptors<rascal::StructureManagerLammps>(manager, adaptors);
  std::cout << "Sucessfully created managers" << std::endl;
  //for (auto atom : manager) {
  //  std::cout << "atom " << atom.get_atom_tag() << " global index "
  //            << atom.get_global_index() << std::endl;
  //}


  //auto managers{rascal::ManagerCollection<
  //                                 rascal::StructureManagerLammps,
  //                                 rascal::AdaptorNeighbourList,
  //                                   rascal::AdaptorCenterContribution
  //                                 rascal::AdaptorStrict>(adaptors)};
  //managers
  std::cout << "PairRASCAL::compute end" << std::endl;
}

// I think rascal also require in metal units, since QUIP does
//  QUIP potentials are parameterized in electron-volts and Angstroms and therefore should be used with LAMMPS metal units.
// https://docs.lammps.org/pair_quip.html

void PairRASCAL::settings(int narg, char ** /* arg */)
{
  std::cout << "PairRASCAL::settings start" << std::endl;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");

  // QUIP potentials are parameterized in metal units
  // TODO also rascal?
  if (strcmp("metal",update->unit_style) != 0)
    error->all(FLERR,"QUIP potentials require 'metal' units");
  std::cout << "PairRASCAL::settings end" << std::endl;
}

void PairRASCAL::allocate()
{
  std::cout << "PairRASCAL::allocate start" << std::endl;
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag,n+1,n+1,"pair:setflag");
  cutsq = memory->create(cutsq,n+1,n+1,"pair:cutsq");
  map = new int[n+1];
  std::cout << "PairRASCAL::allocate end" << std::endl;
}

// For initialization, gives the input of of pair_coeff
// e.g. If your file 
// pair_style      quip
// pair_coeff      * * gap_example.xml "Potential xml_label=GAP_2014_5_8_60_17_10_38_466" 14
// arg contains the arguments ['*', '*', 'gap_example', 'Potential xml_label=GAP_2014_5_8_60_17_10_38_466', '14']
void PairRASCAL::coeff(int narg, char **arg)
{
  std::cout << "PairRASCAL::coeff start" << std::endl;
  if (!allocated) allocate();

  //int n = atom->ntypes;
  if (narg != (3))
    error->all(FLERR,"Number of arguments {} is not correct, "
                                 "it should be {}", narg, 3);

  // ensure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  rascal_file = utils::strdup(arg[2]);
  // needed?
  n_rascal_file = strlen(rascal_file);
  //quip_string = utils::strdup(arg[3]);
  //n_quip_file = strlen(quip_file);
  //n_quip_string = strlen(quip_string);

  // here we load model file
  //json input{read_json(rascal_file)}; // check this line
  //json adaptors_input = input.at("adaptors").template get<json>();
  //json calculator_input = input.at("calculator").template get<json>();
  //json kernel_input = input.at("kernel").template get<json>();
  //auto selected_ids = input.at("selected_ids")
  //                        .template get<std::vector<std::vector<int>>>();

  //// rascal initalize model
  //Kernel_t kernel{kernel_input};
  //kernel_input.at("target_type") = "Atom";
  //Kernel_t kernel_num{kernel_input};
  //SparsePoints_t sparse_points{};
  //Representation_t representation{calculator_input};
  //// load structures, compute representation and fill sparse points
  //// This requires an equivalent for StructureManagerLammps input 
  ////managers.add_structures(filename, 0,
  ////                        input.at("n_structures").template get<int>());
  //// requires fnu
  ////sparse_points.push_back(representation, managers, selected_ids);
  //calculator_input["compute_gradients"] = false;
  //Representation_t representation_{calculator_input};

  // ManagerCollection_t managers{adaptors_input};
  std::cout << "PairRASCAL::coeff end" << std::endl;
}

void PairRASCAL::init_style()
{
  std::cout << "PairRASCAL::init_style start" << std::endl;
  // TODO check if necessary
  // Require newton pair on
  if (force->newton_pair != 1)
    error->all(FLERR,"Pair style quip requires newton pair on");

  // Initialise neighbor list
  int irequest_full = neighbor->request(this);
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  std::cout << "PairRASCAL::init_style end" << std::endl;
}

double PairRASCAL::init_one(int /*i*/, int /*j*/)
{
  std::cout << "PairRASCAL::init_one start" << std::endl;
  return 5;
  std::cout << "PairRASCAL::init_one end" << std::endl;
}
