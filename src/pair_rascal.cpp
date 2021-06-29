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

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *type = atom->type;
  // TODO(alex) should be properly handeled
  // tagint can be also int or int64_t depending on compile flags see LAMMPS_BIGBIG
  int *tag = atom->tag;
  //std::cout << "sizeof(tagint)" << sizeof(tagint) << std::endl;
  //for (int i{0}; i < ntotal; i++) {
  //  std::cout << "atom->tag[i] " << (int) atom->tag[i] << std::endl;
  //}

  double **x = atom->x;
  double **f = atom->f;

  int const tot_num = atom->nlocal + atom->nghost;

  // seems to be lammps intern
  ev_init(eflag,vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  std::cout << "list->ghost " << list->ghost << std::endl;
  //for (int i{0}; i < nlocal; i++) {
  //  for (int j{0}; j < numneigh[i]; j++) {
  //    std::cout << "ilist[i] (" << ilist[i] << ", " << firstneigh[i][j] << ")" << std::endl;
  //  }
  //}


  //   int inum, int tot_num, int * ilist,
  //   int * numneigh, int ** firstneigh,
  //   double ** x, double ** f, int * type,
  //   double * eatom, double ** vatom
  //std::cout << "Types : " << type[0] << " " << type[1] << std::endl;
  //std::cout << "Creating root manager..." << std::endl;
  //std::cout << "Created root manager." << std::endl;
  //// & ? needed somewhere
  //std::cout << "Update root manager..." << std::endl;
  //std::cout << "rascal_atom_types.size() " << rascal_atom_types.size() << std::endl;
  //auto root_manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();
  //root_manager->update(inum, tot_num, ilist, numneigh, firstneigh, x, f, type, eatom, vatom, rascal_atom_types, tag);
  //std::cout << "Updated root manager." << std::endl;
  //for (auto atom : root_manager->with_ghosts()) {
  //  std::cout << "atom " << atom.get_atom_tag() << " global index "
  //            << atom.get_global_index() << std::endl;
  //}
  ////for (auto atom : root_manager) {
  ////  for (auto pair : atom.pairs()) {
  ////    std::cout << "pair " << atom.get_atom_tag() << " " << pair.get_atom_tag() << std::endl;
  ////  }
  ////}
  //std::cout << std::endl;

  //json adaptors_input = {
  //    {
  //      {"initialization_arguments", {}},
  //      {"name", "centercontribution"},
  //    },
  //    {
  //      {"initialization_arguments", {{"cutoff", cutoff}}},
  //      {"name", "strict"}
  //    }
  //};
  //// make ManagerCollection to use with sparse kernel predict functions
  //auto manager = rascal::stack_adaptors<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(root_manager, adaptors_input);
  ////auto manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();
  //auto managers{rascal::ManagerCollection<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(adaptors_input)};
  //managers.add_structure(manager);
  std::shared_ptr<rascal::AdaptorStrict<rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>> manager = *managers.begin();
  manager->update(inum, tot_num, ilist, numneigh, firstneigh, x, f, type, eatom, vatom, rascal_atom_types, tag);
  for (auto atom : manager->with_ghosts()) {
      std::cout << " center " << atom.get_atom_tag() << std::endl;
    for (auto pair : atom.pairs_with_self_pair()) {
      std::cout << "pair (" << atom.get_atom_tag() << ", "
                << pair.get_atom_tag() << ") global index "
                << pair.get_global_index() << std::endl;
    }
  }




  std::cout << "Computing representation..." << std::endl;
  calculator->compute(managers);
  std::cout << "Computed representation." << std::endl;

  //// predict gradient, stress

  ////std::cout << "weights_vec " << std::endl;
  ////std::cout << weights_vec.size() << std::endl;
  ////std::cout << std::endl;
  ////std::cout << "weights " << std::endl;
  ////std::cout << weights << std::endl;
  ////// manual
  rascal::math::Matrix_t KNM{kernel->compute(*calculator, managers, sparse_points)};
  std::cout << "KNM shape " << KNM.rows() << ", " << KNM.cols() << std::endl;
  std::cout << "weights shape " << weights.rows() << ", " << weights.cols() << std::endl;
  rascal::math::Matrix_t energies = KNM * weights.transpose();

  std::cout << "compute_sparse_kernel_gradients" << std::endl;
  std::string force_name = rascal::compute_sparse_kernel_gradients(
          *calculator, *kernel, managers, sparse_points, weights);

  // needs some small adaptation
  //std::string neg_stress_name = rascal::compute_sparse_kernel_neg_stress(
  //    *calculator, *kernel, managers, sparse_points, weights);

  std::cout << "get gradients" << std::endl;
  auto && gradients{*manager->template get_property<
      rascal::Property<double, 1, rascal::AdaptorStrict<
   rascal::AdaptorCenterContribution<rascal::StructureManagerLammps>>, 1, 3>>(force_name, true)};

  std::cout << "matrix map" << std::endl;
  rascal::math::Matrix_t rascal_force = Eigen::Map<const rascal::math::Matrix_t>(
       gradients.view().data(), manager->size(), 3);
  std::cout << "rascal_force" << std::endl;
  std::cout << rascal_force << std::endl;
  std::cout <<  "nlocal + nghost = " << nlocal << "+" << nghost <<std::endl;
  for (ii = 0; ii < nlocal; ii++) {
     for (jj = 0; jj < 3; jj++) {
        f[ii][jj] +=  rascal_force(ii, jj);
     }
  }
  
  // TODO(alex) not sure
  //if (eflag_global) {
  //  eng_vdwl = quip_energy;
  //}

  if (eflag_atom) {
    for (ii = 0; ii < nlocal; ii++) {
      eatom[ii] = energies(ii);
    }
  }
  
  //size_t i_center{0};
  //for (auto manager : managers) {
  //  rascal::math::Matrix_t ee =
  //      energies.block(i_center, 0, 1, 1);
  //  std::cout << "ee shape: " << ee.rows() << ", " << ee.cols() << std::endl;

  //  auto && gradients{*manager->template get_property<
  //      Property<double, 1, Manager_t, 1, ThreeD>>(force_name, true)};
  //  rascal::math::Matrix_t ff = Eigen::Map<const math::Matrix_t>(
  //      gradients.view().data(), manager->size() * ThreeD, 1);
  //  std::cout << "ff shape: " << ff.rows() << ", " << ff.cols() << std::endl;

  //  auto && neg_stress{
  //      *manager->template get_property<Property<double, 0, Manager_t, 6>>(
  //          neg_stress_name, true)};
  //  rascal::math::Matrix_t ff_stress =
  //     Eigen::Map<const math::Matrix_t>(neg_stress.view().data(), 6, 1);
  //  std::cout << "ff_stress shape: " << ff_stress.rows() << ", " << ff_stress.cols() << std::endl;

  //  i_center += manager->size() * ThreeD;
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

  int n = atom->ntypes;
  if (narg != (3 + n))
    error->all(FLERR,"Number of arguments {} is not correct, "
                                 "it should be {}", narg, 3 + n);

  // ensure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  rascal_file = utils::strdup(arg[2]);
  rascal_atom_types.reserve(n);
  for (int i=0; i<n; i++) {
    rascal_atom_types.emplace_back(std::atoi(arg[i+3]));
    std::cout << "rascal_atom_types " << rascal_atom_types[i] << std::endl;
  }


  // needed? TODO(alex)
  n_rascal_file = strlen(rascal_file);
  //quip_string = utils::strdup(arg[3]);
  //n_quip_file = strlen(quip_file);
  //n_quip_string = strlen(quip_string);

  // here we load model file TODO(alex) should be within rascal with a function load_rascal_model(filename)
  json input = rascal::json_io::load(rascal_file);
  std::cout << "input loaded" << std::endl;
  json init_params = input.at("init_params").template get<json>();
  std::cout << "init_params" << std::endl;
  json X_train = init_params.at("X_train").template get<json>();
  std::cout << "X_train" << std::endl;

  // sparse points
  json sparse_data = X_train.at("data").template get<json>();
  json sparse_input = sparse_data.at("sparse_points").template get<json>();
  //sparse_points = rascal::SparsePointsBlockSparse<rascal::CalculatorSphericalInvariants>();
  rascal::from_json(sparse_input, sparse_points);

  // calculator
  json X_train_init_params = X_train.at("init_params").template get<json>();
  json representation = X_train_init_params.at("representation").template get<json>();
  json representation_init_params = representation.at("init_params").template get<json>();
  calculator = std::make_shared<rascal::CalculatorSphericalInvariants>(representation_init_params);

  // kernel
  json kernel_params = init_params.at("kernel").template get<json>();
  json kernel_init_params = kernel_params.at("init_params").template get<json>();
  kernel = std::make_shared<rascal::SparseKernel>(kernel_init_params);

  // weights
  std::vector<double> weights_vec = init_params.at("weights").template get<json>().at(1).template get<std::vector<double>>();
  // TODO(alex) I think does copy, but double check
  weights = Eigen::Map<rascal::math::Vector_t>(weights_vec.data(), static_cast<long int>(weights_vec.size()));

  // cutoff
  cutoff = representation_init_params.at("cutoff_function").template get<json>().at("cutoff").template get<json>().at("value").template get<double>();

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

  auto root_manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();
  //auto neigh_manager{make_adapted_manager<rascal::AdaptorCenterContribution>(strict_manager)};
  //auto strict_manager{make_adapted_manager<rascal::AdaptorStrict>(strict_manager)};

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
  // make ManagerCollection to use with sparse kernel predict functions
  //manager = rascal::stack_adaptors<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(root_manager, adaptors_input);
  //auto manager = rascal::make_structure_manager<rascal::StructureManagerLammps>();
  auto manager = rascal::stack_adaptors<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(root_manager, adaptors_input);
  managers = rascal::ManagerCollection<rascal::StructureManagerLammps, rascal::AdaptorCenterContribution, rascal::AdaptorStrict>(adaptors_input);

  managers.add_structure(manager);

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
  return cutoff;
  std::cout << "PairRASCAL::init_one end" << std::endl;
}
