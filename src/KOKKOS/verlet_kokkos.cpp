// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "verlet_kokkos.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify_kokkos.h"
#include "timer.h"
#include "kokkos.h"

#ifdef LMP_KOKKOS_STDEXEC
#include "Kokkos_exec.hpp"
#include <stdexec/execution.hpp>
#include <exception>
#include <utility>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#endif

using namespace LAMMPS_NS;

namespace {
template<class ViewA, class ViewB>
struct ForceAdder {
  ViewA a;
  ViewB b;
  ForceAdder(const ViewA& a_, const ViewB& b_):a(a_),b(b_) {}
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    a(i,0) += b(i,0);
    a(i,1) += b(i,1);
    a(i,2) += b(i,2);
  }
};
}    // namespace

/* ---------------------------------------------------------------------- */

namespace {
template<class View>
struct Zero {
  View v;
  Zero(const View &v_):v(v_) {}
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const {
    v(i,0) = 0;
    v(i,1) = 0;
    v(i,2) = 0;
  }
};
}    // namespace

/* ----------------------------------------------------------------------
   zero count entries of a per-atom array on both host sides, from first

   the loops that zero the device copy leave the host side alone.  A style
   that runs on the host adds its forces into the Kokkos host view, and a
   style without KOKKOS support adds into the plain LAMMPS array behind it,
   so whatever either of them added last step is still there and is counted
   again when the two sides are brought together.  Each call takes the same
   range as the device loop it goes with, so that the atoms the plain code
   leaves alone -- those outside an include group -- are left alone here too.

   needed says whether any force style runs on the host side at all, see
   host_force_styles().  When none does, nothing accumulates into the host
   copies and nothing reads them without syncing them from the (zeroed)
   device copy first, so the pass over them is skipped: on a GPU it would
   otherwise write the whole host force array twice on every step.
------------------------------------------------------------------------- */

template<class View>
static void zero_host_view(const View &v, int first, int count)
{
  Kokkos::parallel_for(Kokkos::RangePolicy<LMPHostType>(first,first+count),Zero<View>(v));
}

template<class DualView>
static void zero_host(const DualView &k, int first, int count, int needed)
{
  if (!needed || (count <= 0)) return;
  zero_host_view(k.view_hostkk(),first,count);
  zero_host_view(k.view_host(),first,count);
}

#ifdef LMP_KOKKOS_STDEXEC
namespace {

// A sender cannot be started without a receiver as connect() needs one to
// build the operation state, and the receiver is where completion is
// delivered. This one ignores the value and stopped channels. The Kokkos
// sender completes inline, inside start(), as soon as the kernel is launched.
// We dont want to use stdexec::sync_wait as it fences the
// execution space, which is what we want to avoid.

struct LaunchReceiver {
  using receiver_concept = stdexec::receiver_t;
  std::exception_ptr *error;

  template <class... Args>
  void set_value(Args &&...) noexcept {}

  template <class E>
  void set_error(E &&err) noexcept {
    try {
      if constexpr (std::is_same_v<std::decay_t<E>, std::exception_ptr>)
        *error = std::forward<E>(err);
      else
        throw std::forward<E>(err);
    } catch (...) {
      *error = std::current_exception();
    }
  }

  void set_stopped() noexcept {}

  auto get_env() const noexcept -> stdexec::env<> { return {}; }
};

// We bind the sender here to the LaunchReceiver, get an op state and start the work.
// For the Kokkos stream senders used here this launches the kernel on the scheduler's
//  stream and completes the receiver inline. Here completion means kernel launched. The kernel is 
// likely still running when this returns. 
// 
// Later steps are ordered after it with events.
// Note that the operation state lives on the stack and is destroyed on return. That is
// only valid because the sender has already completed inside start(). A sender
// that completes asynchronously (e.g. after continues_on) would need
// stdexec::start_detached() or an operation state that outlives this call.
// An error delivered to the receiver is rethrown here.
template <class Sender>
void start_sender(Sender &&snd)
{
  std::exception_ptr error;
  auto operation = stdexec::connect(std::forward<Sender>(snd),
                                     LaunchReceiver{&error});
  stdexec::start(operation);
  if (error) std::rethrow_exception(error);
}

// Scheduler used to zero the per-atom force arrays (f, and torque and the
// SPIN forces when present).
//
// On CUDA this is a side stream, so the zeros overlap the default-stream
// forward communication (the ghost position pack/MPI/unpack, or the
// on-device ghost copy on one rank). All arrays are zeroed on this one
// stream, so they run in order and a single order_streams(DefaultAfterZero) covers all of
// them. The pair kernel stays on the default stream and waits for this stream
// on the device.
// Without CUDA this is a scheduler on the default instance.
// The scheduler (and its stream) is a function-local static, created on
// first use and kept for the lifetime of the process.
Kokkos::Experimental::exec::scheduler<LMPDeviceType> &zero_scheduler()
{
#ifdef KOKKOS_ENABLE_CUDA
  static auto parts = Kokkos::Experimental::partition_space(LMPDeviceType{}, 1);
  static Kokkos::Experimental::exec::scheduler<LMPDeviceType> sched{parts[0]};
  return sched;
#else
  static Kokkos::Experimental::exec::scheduler<LMPDeviceType> gpu{};
  return gpu;
#endif
}

#ifdef KOKKOS_ENABLE_CUDA
// Which stream waits for which, see order_streams().
enum class ZeroOrder {
  ZeroAfterDefault,    // zero stream waits for the default stream
  DefaultAfterZero     // default stream waits for the zero stream
};

// Order the zero stream and the default stream on the device, without
// blocking the host.
//
// An event is recorded on the stream that goes first and the other stream is
// told to wait for it (cudaStreamWaitEvent). Both calls return immediately,
// the ordering is enforced on the device. The wait captures the event as
// recorded at that point, so one event serves both directions.
//
// ZeroAfterDefault: the integrate kernels on the default stream read the
// forces and the zeros on the side stream write them, so the zeros must not
// start before those kernels finish. Call this before launching zeros.
//
// DefaultAfterZero: the pair kernel, which runs on the default stream, starts
// only after all zeroed arrays (f, torque, SPIN forces) are done, as they
// share the zero stream.
void order_streams(ZeroOrder order)
{
  auto *main = LMPDeviceType{}.impl_internal_space_instance();
  auto *side = zero_scheduler().execution_space().impl_internal_space_instance();
  auto *first = (order == ZeroOrder::ZeroAfterDefault) ? main : side;
  auto *second = (order == ZeroOrder::ZeroAfterDefault) ? side : main;

  static cudaEvent_t event = nullptr;
  if (!event)
    main->cuda_event_create_with_flags_wrapper(&event, cudaEventDisableTiming);
  first->cuda_event_record_wrapper(event);
  second->set_cuda_device();
  cudaStreamWaitEvent(second->m_stream, event, 0);
}
#endif

}
#endif

/* ----------------------------------------------------------------------
   zero rows begin..end-1 of a per-atom 3-vector device view

   with stdexec the zero is launched as a sender on the zero scheduler (see
   zero_scheduler()) and the call returns after the launch. Without stdexec
   this is a plain Kokkos::parallel_for on the default instance. Neither
   variant waits for the kernel to finish.
------------------------------------------------------------------------- */

template<class View>
static void parallel_zero(int begin, int end, const View &v)
{
  auto policy = Kokkos::RangePolicy<LMPDeviceType>(begin, end);
#ifdef LMP_KOKKOS_STDEXEC
  namespace ex = stdexec;
  namespace kex = Kokkos::Experimental::exec;
  start_sender(ex::schedule(zero_scheduler()) |
               kex::parallel_for(policy, Zero<View>(v)));
#else
  Kokkos::parallel_for(policy, Zero<View>(v));
#endif
}

/* ---------------------------------------------------------------------- */

VerletKokkos::VerletKokkos(LAMMPS *lmp, int narg, char **arg) :
  Verlet(lmp, narg, arg)
{
  atomKK = (AtomKokkos *) atom;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void VerletKokkos::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fputs("Setting up Verlet run ...\n",screen);
#ifdef LMP_KOKKOS_STDEXEC
    fputs("  Kokkos stdexec: force zero overlaps comm; pair waits on device\n",screen);
#endif
    if (flag) {
      utils::print(screen,"  Unit style    : {}\n"
                        "  Current step  : {}\n"
                        "  Time step     : {}\n",
                 update->unit_style,update->ntimestep,update->dt);
      timer->print_timeout(screen);
    }
  }

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  lmp->kokkos->auto_sync = 1;

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) {
    // a kokkosable pair claims the arrays it writes (e.g. F) just before its
    // kernel and relies on that claim surviving until the kernel runs.  the
    // setup path otherwise leaves auto_sync on, which syncs each claim straight
    // back to the host and drops it, so the device force the kernel writes
    // never reaches the host and later styles read stale data.  disable
    // auto_sync for a kokkosable pair here, as the run loop already does
    int prev_auto_sync = lmp->kokkos->auto_sync;
    if (force->pair->kokkosable) lmp->kokkos->auto_sync = 0;
    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    force->pair->compute(eflag,vflag);
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);
    lmp->kokkos->auto_sync = prev_auto_sync;
  } else if (force->pair) force->pair->compute_dummy(eflag,vflag,0);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) {
      atomKK->sync(force->bond->execution_space,force->bond->datamask_read);
      force->bond->compute(eflag,vflag);
      atomKK->modified(force->bond->execution_space,force->bond->datamask_modify);
    }
    if (force->angle) {
      atomKK->sync(force->angle->execution_space,force->angle->datamask_read);
      force->angle->compute(eflag,vflag);
      atomKK->modified(force->angle->execution_space,force->angle->datamask_modify);
    }
    if (force->dihedral) {
      atomKK->sync(force->dihedral->execution_space,force->dihedral->datamask_read);
      force->dihedral->compute(eflag,vflag);
      atomKK->modified(force->dihedral->execution_space,force->dihedral->datamask_modify);
    }
    if (force->improper) {
      atomKK->sync(force->improper->execution_space,force->improper->datamask_read);
      force->improper->compute(eflag,vflag);
      atomKK->modified(force->improper->execution_space,force->improper->datamask_modify);
    }
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) {
      atomKK->sync(force->kspace->execution_space,force->kspace->datamask_read);
      force->kspace->compute(eflag,vflag);
      atomKK->modified(force->kspace->execution_space,force->kspace->datamask_modify);
    } else force->kspace->compute_dummy(eflag,vflag,0);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  lmp->kokkos->auto_sync = 0;
  modify->setup(vflag);
  lmp->kokkos->auto_sync = 1;

  atomKK->sync(Host,ALL_MASK);
  output->setup(flag);
  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void VerletKokkos::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  lmp->kokkos->auto_sync = 1;

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) {
    // a kokkosable pair claims the arrays it writes (e.g. F) just before its
    // kernel and relies on that claim surviving until the kernel runs.  the
    // setup path otherwise leaves auto_sync on, which syncs each claim straight
    // back to the host and drops it, so the device force the kernel writes
    // never reaches the host and later styles read stale data.  disable
    // auto_sync for a kokkosable pair here, as the run loop already does
    int prev_auto_sync = lmp->kokkos->auto_sync;
    if (force->pair->kokkosable) lmp->kokkos->auto_sync = 0;
    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    force->pair->compute(eflag,vflag);
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);
    lmp->kokkos->auto_sync = prev_auto_sync;
  } else if (force->pair) force->pair->compute_dummy(eflag,vflag,0);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) {
      atomKK->sync(force->bond->execution_space,force->bond->datamask_read);
      force->bond->compute(eflag,vflag);
      atomKK->modified(force->bond->execution_space,force->bond->datamask_modify);
    }
    if (force->angle) {
      atomKK->sync(force->angle->execution_space,force->angle->datamask_read);
      force->angle->compute(eflag,vflag);
      atomKK->modified(force->angle->execution_space,force->angle->datamask_modify);
    }
    if (force->dihedral) {
      atomKK->sync(force->dihedral->execution_space,force->dihedral->datamask_read);
      force->dihedral->compute(eflag,vflag);
      atomKK->modified(force->dihedral->execution_space,force->dihedral->datamask_modify);
    }
    if (force->improper) {
      atomKK->sync(force->improper->execution_space,force->improper->datamask_read);
      force->improper->compute(eflag,vflag);
      atomKK->modified(force->improper->execution_space,force->improper->datamask_modify);
    }
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) {
      atomKK->sync(force->kspace->execution_space,force->kspace->datamask_read);
      force->kspace->compute(eflag,vflag);
      atomKK->modified(force->kspace->execution_space,force->kspace->datamask_modify);
    } else force->kspace->compute_dummy(eflag,vflag,0);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  lmp->kokkos->auto_sync = 0;
  modify->setup(vflag);
  lmp->kokkos->auto_sync = 1;

  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void VerletKokkos::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force = modify->n_post_force_any;
  int n_end_of_step = modify->n_end_of_step;

  lmp->kokkos->auto_sync = 0;

  fuse_integrate = 0;
  fuse_force_clear = 0;

  if (atomKK->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  f_merge_copy = DAT::t_kkacc_1d_3("VerletKokkos::f_merge_copy",atomKK->k_f.extent(0));

  atomKK->sync(Device,ALL_MASK);

  timer->init_timeout();
  for (int i = 0; i < n; i++) {
    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);

    // initial time integration

    timer->stamp();
    if (!fuse_integrate)
      modify->initial_integrate(vflag);
    if (n_post_integrate) modify->post_integrate();
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

#ifdef LMP_KOKKOS_STDEXEC
#ifdef KOKKOS_ENABLE_CUDA
    // No reallocation on this path. Zeros run on a side stream while
    // forward_comm fences only the default instance.
    if (nflag == 0) clear_force_arrays(0);
#endif
#endif

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
        timer->stamp();
        modify->pre_exchange();
        timer->stamp(Timer::MODIFY);
      }
      if (triclinic) domain->x2lamda(atomKK->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();

      comm->exchange();
      if (sortflag && ntimestep >= atomKK->nextsort) atomKK->sort();
      comm->borders();

      if (triclinic) domain->lamda2x(atomKK->nlocal+atomKK->nghost);

      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) {
        modify->pre_neighbor();
        timer->stamp(Timer::MODIFY);
      }
      neighbor->build(1);
      timer->stamp(Timer::NEIGH);
      if (n_post_neighbor) {
        modify->post_neighbor();
        timer->stamp(Timer::MODIFY);
      }
    }

    // check if kernels can be fused, must come after initial_integrate

    fuse_check(i,n);

    // force computations
    // important for pair to come before bonded contributions
    // since some bonded potentials tally pairwise energy/virial
    // and Pair:ev_tally() needs to be called before any tallying

#ifdef LMP_KOKKOS_STDEXEC
#ifdef KOKKOS_ENABLE_CUDA
    if (nflag != 0) clear_force_arrays(0);
    // Pair stream waits for the force zero. The host does not.
    if (!external_force_clear) order_streams(ZeroOrder::DefaultAfterZero);
#else
    clear_force_arrays(0);
#endif
#else
    if (!fuse_force_clear)
      clear_force_arrays(0);
#endif

    timer->stamp();

    if (n_pre_force) {
      modify->pre_force(vflag);
      timer->stamp(Timer::MODIFY);
    }

    bool execute_on_host = false;
    uint64_t datamask_read_host = 0;
    uint64_t datamask_exclude = 0;

    // host_force_styles() is 0 without overlap, so this also tells whether
    // the host force copies are in use

    execute_on_host = host_force_styles(&datamask_read_host);

    // exclude the forces only when they are merged below

    if (execute_on_host) datamask_exclude = (F_MASK | ENERGY_MASK | VIRIAL_MASK);

    // sync what the pre_force fixes claimed on the forces before excluding them,
    // to where the pair accumulates, as the pair's own sync below would

    if (execute_on_host)
      atomKK->sync(pair_compute_flag ? force->pair->execution_space : Device, datamask_exclude);

    // keep the forces out of sync() and modified() until they are merged

    AtomKokkos::ExcludeMask exclude_guard(atomKK,datamask_exclude);

    // when a non-KOKKOS style runs inside a KOKKOS run, enable auto_sync for
    // the duration of its compute so that any sync()/modified() it triggers
    // (e.g. via the DomainKokkos x2lamda/lamda2x overrides) writes changes
    // through to the legacy host arrays the style reads and writes

    if (pair_compute_flag) {
      int prev_auto_sync = lmp->kokkos->auto_sync;
      if (!force->pair->kokkosable) lmp->kokkos->auto_sync = 1;
      atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
      force->pair->compute(eflag,vflag);
      lmp->kokkos->auto_sync = prev_auto_sync;
      atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);
      timer->stamp(Timer::PAIR);
    }

    if (execute_on_host) {
      if (pair_compute_flag && force->pair->datamask_modify != datamask_exclude)
        Kokkos::fence();
      // sync_pinned() is not routed through the exclude mask, so mask here
      atomKK->sync_pinned(HostKK,~(~datamask_read_host|datamask_exclude),1);
      // zero the host-side force buffer before the host styles accumulate into
      // it, so the later device/host force merge does not re-add stale values.
      // skip this only when the pair style itself runs on the host, since then
      // the pair force we want to merge already lives in this buffer.  the old
      // guard required pair_compute_flag, so with the pair disabled (e.g.
      // "pair_modify compute no") but host-side bonded styles still present the
      // buffer was left stale and its contents were re-added every step.
      if (!pair_compute_flag || (force->pair->execution_space != HostKK &&
          force->pair->execution_space != Host)) {
        Kokkos::deep_copy(LMPHostType(),atomKK->k_f.view_hostkk(),0.0);
        atomKK->k_f.modify_hostkk_legacy();

        // a force style without KOKKOS support (execution_space == Host) adds
        // into the legacy host array, which is a separate allocation whenever
        // the two host views need a transform, so it has to be cleared for the
        // same reason.  Without this, fusing force_clear() into the pair style
        // leaves it stale and its contents are re-added on every step.

        if (decltype(atomKK->k_f)::NEED_TRANSFORM)
          Kokkos::deep_copy(LMPHostType(),atomKK->k_f.view_host(),0.0);
      }
    }

    if (atomKK->molecular) {
      if (force->bond) {
        int prev_auto_sync = lmp->kokkos->auto_sync;
        if (!force->bond->kokkosable) lmp->kokkos->auto_sync = 1;
        atomKK->sync(force->bond->execution_space,force->bond->datamask_read);
        force->bond->compute(eflag,vflag);
        lmp->kokkos->auto_sync = prev_auto_sync;
        atomKK->modified(force->bond->execution_space,force->bond->datamask_modify);
      }
      if (force->angle) {
        int prev_auto_sync = lmp->kokkos->auto_sync;
        if (!force->angle->kokkosable) lmp->kokkos->auto_sync = 1;
        atomKK->sync(force->angle->execution_space,force->angle->datamask_read);
        force->angle->compute(eflag,vflag);
        lmp->kokkos->auto_sync = prev_auto_sync;
        atomKK->modified(force->angle->execution_space,force->angle->datamask_modify);
      }
      if (force->dihedral) {
        int prev_auto_sync = lmp->kokkos->auto_sync;
        if (!force->dihedral->kokkosable) lmp->kokkos->auto_sync = 1;
        atomKK->sync(force->dihedral->execution_space,force->dihedral->datamask_read);
        force->dihedral->compute(eflag,vflag);
        lmp->kokkos->auto_sync = prev_auto_sync;
        atomKK->modified(force->dihedral->execution_space,force->dihedral->datamask_modify);
      }
      if (force->improper) {
        int prev_auto_sync = lmp->kokkos->auto_sync;
        if (!force->improper->kokkosable) lmp->kokkos->auto_sync = 1;
        atomKK->sync(force->improper->execution_space,force->improper->datamask_read);
        force->improper->compute(eflag,vflag);
        lmp->kokkos->auto_sync = prev_auto_sync;
        atomKK->modified(force->improper->execution_space,force->improper->datamask_modify);
      }
      timer->stamp(Timer::BOND);
    }

    if (kspace_compute_flag) {
      int prev_auto_sync = lmp->kokkos->auto_sync;
      if (!force->kspace->kokkosable) lmp->kokkos->auto_sync = 1;
      atomKK->sync(force->kspace->execution_space,force->kspace->datamask_read);
      force->kspace->compute(eflag,vflag);
      lmp->kokkos->auto_sync = prev_auto_sync;
      atomKK->modified(force->kspace->execution_space,force->kspace->datamask_modify);
      timer->stamp(Timer::KSPACE);
    }

    if (execute_on_host) {
      if (f_merge_copy.extent(0) < atomKK->k_f.extent(0))
        f_merge_copy = DAT::t_kkacc_1d_3("VerletKokkos::f_merge_copy",atomKK->k_f.extent(0));
      f = atomKK->k_f.view_device();

      // both host copies can hold a contribution: a /kk/host style accumulates
      // into the Kokkos host view, a style without KOKKOS support into the
      // legacy host array behind atom->f, which is a separate allocation when
      // the two need a transform.  Add the legacy one in.  sync_legacy_to_hostkk()
      // cannot be used here: it would copy one buffer over the other, and it is
      // a no-op anyway since F_MASK is excluded from the modified() calls above.

      if (decltype(atomKK->k_f)::NEED_TRANSFORM) {
        auto h_f_kk = atomKK->k_f.view_hostkk();
        auto h_f_legacy = atomKK->k_f.view_host();
        Kokkos::parallel_for(Kokkos::RangePolicy<LMPHostType>(0,atomKK->k_f.extent(0)),
          ForceAdder<decltype(h_f_kk),decltype(h_f_legacy)>(h_f_kk,h_f_legacy));
      }
      Kokkos::deep_copy(LMPHostType(),f_merge_copy,atomKK->k_f.view_hostkk());
      Kokkos::parallel_for(atomKK->k_f.extent(0),
        ForceAdder<DAT::t_kkacc_1d_3,DAT::t_kkacc_1d_3>(atomKK->k_f.view_device(),f_merge_copy));
      atomKK->k_f.clear_sync_state(); // special case
      atomKK->k_f.modify_device();
    }

    // host and device forces are merged, so sync() and modified() may touch them again

    exclude_guard.release();

    if (n_pre_reverse) {
      modify->pre_reverse(eflag,vflag);
      timer->stamp(Timer::MODIFY);
    }

    // reverse communication of forces

    if (force->newton) {
      Kokkos::fence();
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
#ifdef LMP_KOKKOS_STDEXEC
#ifdef KOKKOS_ENABLE_CUDA
    } else if (torqueflag || extraflag || n_post_force) {
      // The torque zeros use the side stream. Join it before a host reader.
      // The LJ path (newton on, no torque) does not take this branch.
      Kokkos::fence();
#endif
#endif
    }

    // force modifications, final time integration, diagnostics

    if (n_post_force) modify->post_force(vflag);

    if (fuse_integrate) modify->fused_integrate(vflag);
    else modify->final_integrate();

    if (n_end_of_step) modify->end_of_step();
    timer->stamp(Timer::MODIFY);

    // all output

    if (ntimestep == output->next) {
      // non-Kokkos computes and fixes may write through the host pointers,
      // auto_sync carries those writes to the device

      int prev_auto_sync = lmp->kokkos->auto_sync;
      lmp->kokkos->auto_sync = 1;
      atomKK->sync(Host,ALL_MASK);

      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);

      lmp->kokkos->auto_sync = prev_auto_sync;
    }
  }

  atomKK->sync(Host,ALL_MASK);
  lmp->kokkos->auto_sync = 1;
}

/* ----------------------------------------------------------------------
   clear forces, called from setup() and setup_minimal()

   waits for the zero kernels, so the caller can use the forces right away
------------------------------------------------------------------------- */

void VerletKokkos::force_clear()
{
  clear_force_arrays(1);
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   wait = 0 returns after launch so pair compute is the next kernel
   wait = 1 also fences all execution spaces, including the zero stream,
   before returning (stdexec only, otherwise the zero is already ordered)

   with stdexec on CUDA the zeros of all arrays run on one side stream, see
   zero_scheduler(). It is first ordered after the integrate kernels on the
   default stream by order_streams(ZeroAfterDefault). With wait = 0 the caller
   joins the zero stream back into the default stream with
   order_streams(DefaultAfterZero).
   returns without doing anything if the forces are cleared externally
------------------------------------------------------------------------- */

void VerletKokkos::clear_force_arrays(int wait)
{
  if (external_force_clear) return;

#ifdef LMP_KOKKOS_STDEXEC
#ifdef KOKKOS_ENABLE_CUDA
  // Prior kernels on the default stream (integrate) must finish reading
  // forces before these side-stream zeros write them. Device wait, no fence.
  order_streams(ZeroOrder::ZeroAfterDefault);
#endif
#endif

  atomKK->k_f.clear_sync_state(); // ignore host forces/torques since device views
  atomKK->k_torque.clear_sync_state(); //   will be cleared below

  // the SPIN forces below are overwritten in the same way, so their host side
  // has to be released here as well -- without this a host side left claimed
  // from the setup is still claimed when the device side is claimed below, and
  // the two disagree with nothing to say which one is current

  if (extraflag) {
    atomKK->k_fm.clear_sync_state();
    atomKK->k_fm_long.clear_sync_state();
  }

  // the host sides only need clearing when a force style accumulates into
  // them, see zero_host() and host_force_styles()

  const int clear_host = host_force_styles();

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  if (neighbor->includegroup == 0) {
    int nall = atomKK->nlocal;
    if (force->newton) nall += atomKK->nghost;

    parallel_zero(0, nall, atomKK->k_f.view_device());
    zero_host(atomKK->k_f,0,nall,clear_host);
    atomKK->modified(Device,F_MASK);

    if (torqueflag) {
      parallel_zero(0, nall, atomKK->k_torque.view_device());
      zero_host(atomKK->k_torque,0,nall,clear_host);
      atomKK->modified(Device,TORQUE_MASK);
    }

    // reset SPIN forces

    if (extraflag) {
      parallel_zero(0, nall, atomKK->k_fm.view_device());
      zero_host(atomKK->k_fm,0,nall,clear_host);
      atomKK->modified(Device,FM_MASK);
      parallel_zero(0, nall, atomKK->k_fm_long.view_device());
      zero_host(atomKK->k_fm_long,0,nall,clear_host);
      atomKK->modified(Device,FML_MASK);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    int nfirst = atomKK->nfirst;

    parallel_zero(0, nfirst, atomKK->k_f.view_device());
    zero_host(atomKK->k_f,0,nfirst,clear_host);
    atomKK->modified(Device,F_MASK);

    if (torqueflag) {
      parallel_zero(0, nfirst, atomKK->k_torque.view_device());
      zero_host(atomKK->k_torque,0,nfirst,clear_host);
      atomKK->modified(Device,TORQUE_MASK);
    }

    // reset SPIN forces

    if (extraflag) {
      parallel_zero(0, nfirst, atomKK->k_fm.view_device());
      zero_host(atomKK->k_fm,0,nfirst,clear_host);
      atomKK->modified(Device,FM_MASK);
      parallel_zero(0, nfirst, atomKK->k_fm_long.view_device());
      zero_host(atomKK->k_fm_long,0,nfirst,clear_host);
      atomKK->modified(Device,FML_MASK);
    }

    if (force->newton) {
      int begin = atomKK->nlocal;
      int end = begin + atomKK->nghost;

      parallel_zero(begin, end, atomKK->k_f.view_device());
      zero_host(atomKK->k_f,begin,atomKK->nghost,clear_host);
      atomKK->modified(Device,F_MASK);

      if (torqueflag) {
        parallel_zero(begin, end, atomKK->k_torque.view_device());
        zero_host(atomKK->k_torque,begin,atomKK->nghost,clear_host);
        atomKK->modified(Device,TORQUE_MASK);
      }

      if (extraflag) {
        parallel_zero(begin, end, atomKK->k_fm.view_device());
        zero_host(atomKK->k_fm,begin,atomKK->nghost,clear_host);
        atomKK->modified(Device,FM_MASK);
        parallel_zero(begin, end, atomKK->k_fm_long.view_device());
        zero_host(atomKK->k_fm_long,begin,atomKK->nghost,clear_host);
        atomKK->modified(Device,FML_MASK);
      }
    }
  }

#ifdef LMP_KOKKOS_STDEXEC
  if (wait) Kokkos::fence();
#else
  (void) wait;
#endif
}

/* ----------------------------------------------------------------------
   can the force computation overlap host and device work at all?

   only when it is allowed and the two sides have separate memory; in a
   build without a device backend the views alias and there is nothing
   to overlap or to merge
------------------------------------------------------------------------- */

int VerletKokkos::overlap_possible()
{
  if (!lmp->kokkos->allow_overlap) return 0;
  return atomKK->k_f.view_hostkk().data() != atomKK->k_f.view_device().data();
}

/* ----------------------------------------------------------------------
   does any force style run on the host side?

   returns 1 when at least one of the pair, bonded and kspace styles has
   Host or HostKK as its execution space and overlap_possible() holds.
   Those styles accumulate their forces into the host copies of the force
   array, which run() merges into the device copy after the force
   computation, so the host copies are live only in that case.  When a
   mask pointer is given, what those styles read is ORed into it.
------------------------------------------------------------------------- */

int VerletKokkos::host_force_styles(uint64_t *datamask_read_host)
{
  if (!overlap_possible()) return 0;

  int flag = 0;
  auto check = [&](auto *style) {
    if (!style) return;
    if ((style->execution_space != HostKK) && (style->execution_space != Host)) return;
    flag = 1;
    if (datamask_read_host) *datamask_read_host |= style->datamask_read;
  };

  if (pair_compute_flag) check(force->pair);
  if (atomKK->molecular) {
    check(force->bond);
    check(force->angle);
    check(force->dihedral);
    check(force->improper);
  }
  if (kspace_compute_flag) check(force->kspace);

  return flag;
}

/* ----------------------------------------------------------------------
   check if can fuse force_clear() with pair compute()
   Requirements:
   - no pre_force fixes
   - no torques, SPIN forces, or includegroup set
   - pair compute() must be called
   - pair_style must support fusing

   check if can fuse initial_integrate() with final_integrate()
   Requirements:
   - no end_of_step fixes
   - not on last or output step
   - no timers to break out of loop
   - integrate fix style must support fusing
------------------------------------------------------------------------- */

void VerletKokkos::fuse_check(int i, int n)
{
  fuse_force_clear = 1;
  if (modify->n_pre_force) fuse_force_clear = 0;
  else if (torqueflag || extraflag || neighbor->includegroup) fuse_force_clear = 0;
  else if (!force->pair || !pair_compute_flag) fuse_force_clear = 0;
  else if (!force->pair->fuse_force_clear_flag) fuse_force_clear = 0;
#ifdef LMP_KOKKOS_STDEXEC
  // Keep the zero as its own launch so pair compute follows it on the device.
  fuse_force_clear = 0;
#endif

  fuse_integrate = 1;
  if (modify->n_end_of_step) fuse_integrate = 0;
  else if (i == n-1) fuse_integrate = 0;
  else if (update->ntimestep == output->next) fuse_integrate = 0;
  else if (timer->has_timeout()) fuse_integrate = 0;
  else if (!((ModifyKokkos*)modify)->check_fuse_integrate()) fuse_integrate = 0;
}
