/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include "kokkos.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

#include <radial_kokkos.h>
#include <y_grad_kokkos.h>
#include <pair_flare_kokkos.h>

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<t_scalar3<F_FLOAT>> {
    KOKKOS_FORCEINLINE_FUNCTION static t_scalar3<F_FLOAT> sum() {
      return t_scalar3<F_FLOAT>();
    }
  };
}

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairFLAREKokkos<DeviceType>::PairFLAREKokkos(LAMMPS *lmp) : PairFLARE(lmp)
{
  respa_enable = 0;


  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType>
PairFLAREKokkos<DeviceType>::~PairFLAREKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    eatom = NULL;
    vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  nall = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  copymode = 1;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  // build short neighbor list

  max_neighs = d_neighbors.extent(1);
  // TODO: check inum/ignum here
  int n_atoms = neighflag == FULL ? inum : inum;

  if ((d_neighbors_short.extent(1) != max_neighs) ||
     (d_neighbors_short.extent(0) != n_atoms)) {
    d_neighbors_short = Kokkos::View<int**,DeviceType>("FLARE::neighbors_short",n_atoms,max_neighs);
  }
  if (d_numneigh_short.extent(0)!=n_atoms)
    d_numneigh_short = Kokkos::View<int*,DeviceType>("FLARE::numneighs_short",n_atoms);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,n_atoms), *this);

#ifdef LMP_KOKKOS_GPU
  int vector_length = 32;
#else
  int vector_length = 8;
#endif

  // precompute basis functions, reduce register usage
  {
    g = View4D(); Y = View4D();
    g = View4D(Kokkos::ViewAllocateWithoutInitializing("FLARE: g"), n_atoms, max_neighs, n_max, 4);
    Y = View4D(Kokkos::ViewAllocateWithoutInitializing("FLARE: Y"), n_atoms, max_neighs, n_harmonics, 4);
    Kokkos::parallel_for("FLARE: R and Y",
        Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                        {0,0}, {inum, max_neighs}),
        *this
    );
  }

  // compute single bond and its gradient
  {
    single_bond = View3D(); single_bond_grad = View5D();
    single_bond = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: single_bond"), n_atoms, n_radial, n_harmonics);
    single_bond_grad = View5D(Kokkos::ViewAllocateWithoutInitializing("FLARE: single_bond_grad"), n_atoms, max_neighs, 3, n_radial, n_harmonics);

    int g_size = ScratchView2D::shmem_size(n_max, 4);
    int Y_size = ScratchView2D::shmem_size(n_harmonics, 4);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagSingleBond>(inum, Kokkos::AUTO(), vector_length).set_scratch_size(
        0, Kokkos::PerThread(g_size + Y_size));
    Kokkos::deep_copy(single_bond, 0.0);
    Kokkos::deep_copy(single_bond_grad, 0.0);
    Kokkos::parallel_for("FLARE: single bond",
        policy,
        *this
    );
    /*
    Kokkos::parallel_for("FLARE: single bond",
        Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, TagSingleBond>(
                        {0,0,0,0}, {inum, max_neighs, n_max, n_harmonics}),
        *this
    );
    */
  }

  // compute B2
  {
    B2 = View2D();
    B2 = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: B2"), inum, n_descriptors);
    Kokkos::parallel_for("FLARE: B2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, TagB2>(
                        {0,0}, {inum, n_descriptors}),
        *this
    );
  }

  // compute beta*B2
  {
    beta_B2 = View2D();
    beta_B2 = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: beta*B2"), inum, n_descriptors);
    Kokkos::parallel_for("FLARE: beta*B2",
        Kokkos::TeamPolicy<DeviceType, TagBetaB2>(inum, Kokkos::AUTO(), vector_length),
        *this
    );
  }

  // compute B2 squared norms and evdwls and w
  {
    B2_norm2s = View1D(); evdwls = View1D(); w = View2D();
    B2_norm2s = View1D(Kokkos::ViewAllocateWithoutInitializing("FLARE: B2_norm2s"), inum);
    evdwls = View1D(Kokkos::ViewAllocateWithoutInitializing("FLARE: evdwls"), inum);
    w = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: w"), inum, n_descriptors);
    Kokkos::parallel_for("FLARE: B2 norm2 evdwl w",
        Kokkos::TeamPolicy<DeviceType, TagNorm2>(inum, Kokkos::AUTO()),
        *this
    );
  }

  // compute u
  {
    u = View3D();
    u = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: u"), inum, n_radial, n_harmonics);
    Kokkos::parallel_for("FLARE: u",
        Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, Tagu>(
                        {0,0,0}, {inum, n_radial, n_harmonics}),
        *this
    );
  }

  // compute partial forces
  {
    partial_forces = View3D();
    partial_forces = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: partial forces"), inum, max_neighs, 3);
    Kokkos::parallel_for("FLARE: partial forces",
        Kokkos::TeamPolicy<DeviceType, TagF>(inum, Kokkos::AUTO(), vector_length),
        *this
    );
  }

  // sum and store total forces, ev_tally
  {
    vscatter = ScatterVType(d_vatom);
    fscatter = ScatterFType(f);
    Kokkos::parallel_reduce("FLARE: total forces, ev_tally",
        Kokkos::TeamPolicy<DeviceType, TagStoreF>(inum, Kokkos::AUTO()),
        *this,
        ev
    );
    Kokkos::Experimental::contribute(d_vatom, vscatter);
    Kokkos::Experimental::contribute(f, fscatter);
  }

  if (evflag)
    ev_all += ev;

  if (eflag_global) eng_vdwl += ev_all.evdwl;
  if (vflag_global) {
    virial[0] += ev_all.v[0];
    virial[1] += ev_all.v[1];
    virial[2] += ev_all.v[2];
    virial[3] += ev_all.v[3];
    virial[4] += ev_all.v[4];
    virial[5] += ev_all.v[5];
  }

  if (eflag_atom) {
    // if (need_dup)
    //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    //if (need_dup)
      //Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;

}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(const int ii, const int jj) const {

  const int i = d_ilist[ii];
  int j = d_neighbors_short(i,jj);
  j &= NEIGHMASK;

  const X_FLOAT delx = x(j,0) - x(i,0);
  const X_FLOAT dely = x(j,1) - x(i,1);
  const X_FLOAT delz = x(j,2) - x(i,2);
  const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

  calculate_radial_kokkos(ii, jj, g, delx, dely, delz, sqrt(rsq), cutoff, n_max);
  get_Y_kokkos(ii, jj, Y, delx, dely, delz, l_max);
  /*
  printf("i = %d, j = %d, Y =", i, j);
  for(int h = 0; h < n_harmonics; h++){
    printf(" %g", Y(jj, h, 0));
  }
  printf("\n");
  */
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagSingleBond, const MemberType team_member) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii];

  const int jnum = d_numneigh_short[i];

  ScratchView2D gscratch(team_member.thread_scratch(0), n_max, 4);
  ScratchView2D Yscratch(team_member.thread_scratch(0), n_harmonics, 4);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, jnum), [&] (int jj){

      int j = d_neighbors_short(i,jj);
      j &= NEIGHMASK;
      int s = type[j] - 1;


      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, 4*n_max), [&] (int nc){
          int n = nc / 4;
          int c = nc - 4*n;
          gscratch(n, c) = g(ii, jj, n, c);
      });
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, 4*n_harmonics), [&] (int lmc){
          int lm = lmc / 4;
          int c = lmc - 4*lm;
          Yscratch(lm, c) = Y(ii, jj, lm, c);
      });

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, n_max*n_harmonics), [&] (int nlm){
          int n = nlm / n_harmonics;
          int lm = nlm - n_harmonics*n;

          int radial_index = s*n_max + n;
          double g_val = gscratch(n,0);
          double gx_val = gscratch(n,1);
          double gy_val = gscratch(n,2);
          double gz_val = gscratch(n,3);


          double h_val = Yscratch(lm,0);
          double hx_val = Yscratch(lm,1);
          double hy_val = Yscratch(lm,2);
          double hz_val = Yscratch(lm,3);

          double bond = g_val * h_val;
          double bond_x = gx_val * h_val + g_val * hx_val;
          double bond_y = gy_val * h_val + g_val * hy_val;
          double bond_z = gz_val * h_val + g_val * hz_val;

          // Update single bond basis arrays.
          if(jj < jnum) Kokkos::atomic_add(&single_bond(ii, radial_index, lm),bond); // TODO: bad?

          single_bond_grad(ii,jj,0,radial_index,lm) = bond_x;
          single_bond_grad(ii,jj,1,radial_index,lm) = bond_y;
          single_bond_grad(ii,jj,2,radial_index,lm) = bond_z;
      });
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagB2, const int ii, const int nnl) const{
  int x = nnl/(l_max+1);
  int l = nnl-x*(l_max+1);
  double np12 = n_radial + 0.5;
  int n1 = -std::sqrt(np12*np12 - 2*x) + np12;
  int n2 = x - n1*(np12 - 1 - 0.5*n1);

  double tmp = 0.0;
  for(int m = 0; m < 2*l+1; m++){
    int lm = l*l + m;
    tmp += single_bond(ii, n1, lm) * single_bond(ii, n2, lm);
  }
  B2(ii, nnl) = tmp;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagBetaB2, const MemberType team_member) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii];

  const int itype = type[i] - 1;

  // TODO: B2 shared if bottleneck, team-wise GEMV?
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_descriptors), [&] (int &x){
      F_FLOAT tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_descriptors), [&](int &y, F_FLOAT &tmp){
          tmp += beta(itype, x, y)*B2(ii, y);
      }, tmp);
      beta_B2(ii, x) = tmp;
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagNorm2, const MemberType team_member) const{
  int ii = team_member.league_rank();

  F_FLOAT tmp = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x, F_FLOAT &tmp){
      tmp += B2(ii, x) * B2(ii, x);
  }, tmp);
  B2_norm2s(ii) = tmp;

  tmp = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x, F_FLOAT &tmp){
      tmp += B2(ii, x) * beta_B2(ii, x);
  }, tmp);
  evdwls(ii) = tmp/B2_norm2s(ii);
  if (eflag_atom){
    const int i = d_ilist[ii];
    d_eatom[i] = evdwls(ii);
  }

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x){
      w(ii, x) = 2*(evdwls(ii) * B2(ii,x) - beta_B2(ii,x))/B2_norm2s(ii);
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(Tagu, const int ii, const int n1, const int lm) const{
  int l = sqrt(1.0*lm);
  //int l = Kokkos::Experimental::sqrt(lm);

  F_FLOAT un1lm = 0.0;
  for(int n2 = 0; n2 < n_radial; n2++){
    int i = n2 > n1 ? n1 : n2;
    int j = n2 > n1 ? n2 : n1;
    int n1n2 = j + i*(n_radial - 0.5*(i+1));
    int n1n2l = n1n2*(l_max+1)+l;

    un1lm += single_bond(ii, n2, lm) * w(ii, n1n2l) * (1 + (n1 == n2 ? 1 : 0));
  }
  u(ii, n1, lm) = un1lm;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagF, const MemberType team_member) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii];
  const int jnum = d_numneigh_short[i];

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 3*jnum), [&] (int &k){
      int jj = k/3;
      int c = k - 3*jj;
      F_FLOAT tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_bond), [&](int nlm, F_FLOAT &tmp){
          int n = nlm / n_harmonics;
          int lm = nlm - n*n_harmonics;
          tmp += single_bond_grad(ii, jj, c, n, lm)*u(ii, n, lm);
      }, tmp);
      partial_forces(ii,jj,c) = tmp;
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagStoreF, const MemberType team_member, EV_FLOAT &ev) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii];
  const int jnum = d_numneigh_short[i];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);

  auto a_f = fscatter.access();
  t_scalar3<F_FLOAT> fsum;

  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, jnum), [&] (const int jj, t_scalar3<F_FLOAT> &ftmp){
      int j = d_neighbors_short(i,jj);
      j &= NEIGHMASK;

      const F_FLOAT fx = -partial_forces(ii,jj,0);
      const F_FLOAT fy = -partial_forces(ii,jj,1);
      const F_FLOAT fz = -partial_forces(ii,jj,2);

      ftmp.x += fx;
      ftmp.y += fy;
      ftmp.z += fz;

      a_f(j,0) -= fx;
      a_f(j,1) -= fy;
      a_f(j,2) -= fz;

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);

      //printf("i = %d, j = %d, f = %g %g %g\n", i, j, fx, fy, fz);

      if (vflag_either) v_tally(ev,i,j,fx,fy,fz,delx,dely,delz);
  }, fsum);
  team_member.team_barrier();

  Kokkos::single(Kokkos::PerTeam(team_member), [&](){
      a_f(i,0) += fsum.x;
      a_f(i,1) += fsum.y;
      a_f(i,2) += fsum.z;
      if(eflag) ev.evdwl += evdwls(ii);
      //printf("i = %d, Fsum = %g %g %g\n", i, fsum.x, fsum.y, fsum.z);
  });
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(const int& ii) const {
    const int i = d_ilist[ii];
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);

    const int jnum = d_numneigh[i];
    int inside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutoff*cutoff) {
        d_neighbors_short(i,inside) = j;
        inside++;
      }
    }
    d_numneigh_short(i) = inside;
}


/* ---------------------------------------------------------------------- */




/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairFLARE::coeff(narg,arg);

  n_harmonics = (l_max+1)*(l_max+1);
  n_radial = n_species * n_max;
  n_bond = n_radial * n_harmonics;
  n_descriptors = (n_radial * (n_radial + 1) / 2) * (l_max + 1);

  beta = Kokkos::View<F_FLOAT***, Kokkos::LayoutRight, typename DeviceType::memory_space>("beta", n_species, n_descriptors, n_descriptors);
  auto beta_h = Kokkos::create_mirror_view(beta);
  for(int s = 0; s < n_species; s++){
    for(int i = 0; i < n_descriptors; i++){
      for(int j = 0; j < n_descriptors; j++){
        beta_h(s,i,j) = beta_matrices[s](i,j);
      }
    }
  }
  Kokkos::deep_copy(beta, beta_h);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::init_style()
{
  PairFLARE::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<DeviceType,LMPHostType>::value &&
    !std::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<DeviceType,LMPDeviceType>::value;

  // always request a full neighbor list

  //if (neighflag == FULL) { // TODO: figure this out
  //if (neighflag == HALF || neighflag == HALFTHREAD) { // TODO: figure this out
  if (neighflag == FULL || neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
    if (neighflag == FULL)
      neighbor->requests[irequest]->ghost = 1;
    else
      neighbor->requests[irequest]->ghost = 0;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with pair flare/kk");
  }
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::v_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto a_vatom = vscatter.access();

  if (VFLAG) {
    const E_FLOAT v0 = delx*fx;
    const E_FLOAT v1 = dely*fy;
    const E_FLOAT v2 = delz*fz;
    const E_FLOAT v3 = delx*fy;
    const E_FLOAT v4 = delx*fz;
    const E_FLOAT v5 = dely*fz;

    if (vflag_global) {
        ev.v[0] += v0;
        ev.v[1] += v1;
        ev.v[2] += v2;
        ev.v[3] += v3;
        ev.v[4] += v4;
        ev.v[5] += v5;
    }

    if (vflag_atom) {
      a_vatom(i,0) += 0.5*v0;
      a_vatom(i,1) += 0.5*v1;
      a_vatom(i,2) += 0.5*v2;
      a_vatom(i,3) += 0.5*v3;
      a_vatom(i,4) += 0.5*v4;
      a_vatom(i,5) += 0.5*v5;

      a_vatom(j,0) += 0.5*v0;
      a_vatom(j,1) += 0.5*v1;
      a_vatom(j,2) += 0.5*v2;
      a_vatom(j,3) += 0.5*v3;
      a_vatom(j,4) += 0.5*v4;
      a_vatom(j,5) += 0.5*v5;
    }
  }
}



namespace LAMMPS_NS {
template class PairFLAREKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairFLAREKokkos<LMPHostType>;
#endif
}

