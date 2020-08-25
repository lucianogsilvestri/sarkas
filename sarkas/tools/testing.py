"""
Module for testing simulation parameters
"""
# Python modules
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import yaml

# Sarkas modules
from sarkas.utilities.io import InputOutput
from sarkas.utilities.timing import SarkasTimer
from sarkas.potentials.base import Potential
from sarkas.time_evolution.integrators import Integrator
from sarkas.time_evolution.thermostats import Thermostat
from sarkas.base import Particles, Parameters, Species


class PreProcess:
    
    def __init__(self):
        self.potential = Potential()
        self.integrator = Integrator()
        self.thermostat = Thermostat()
        self.parameters = Parameters()
        self.particles = Particles()
        self.species = []
        self.input_file = None
        self.loops = 10
        self.estimate = False
        self.pm_meshes = np.array([16, 24, 32, 40, 48, 56, 64, 80, 112, 128], dtype=int)
        self.pp_cells = np.arange(3, 16, dtype=int)
        self.timer = SarkasTimer()
        self.io = InputOutput()

    def common_parser(self, filename):
        """
        Parse simulation parameters from YAML file.

        Parameters
        ----------
        filename: str
            Input YAML file


        """
        self.input_file = filename
        self.parameters.input_file = filename
        dics = self.io.from_yaml(filename)

        for lkey in dics:
            if lkey == "Particles":
                for species in dics["Particles"]:
                    spec = Species()
                    for key, value in species["Species"].items():
                        if hasattr(spec, key):
                            spec.__dict__[key] = value
                        else:
                            setattr(spec, key, value)
                    self.species.append(spec)

            if lkey == "Potential":
                self.potential.__dict__.update(dics[lkey])

            if lkey == "Thermostat":
                self.thermostat.__dict__.update(dics[lkey])

            if lkey == "Integrator":
                self.integrator.__dict__.update(dics[lkey])

            if lkey == "Parameters":
                self.parameters.__dict__.update(dics[lkey])

            if lkey == "Control":
                self.parameters.__dict__.update(dics[lkey])

    def setup(self, other_inputs=None):
        """
        Initialize all the sub classes of the simulation and save simulation details to log file.

        Parameters
        ----------
        estimate
        loops: int (optional)
            Number of loops over which to average.

        other_inputs : dict (optional)
            Dictionary with additional simulations options.

        """
        if other_inputs:
            if not isinstance(other_inputs, dict):
                raise TypeError("Wrong input type. other_inputs should be a nested dictionary")

            for class_name, class_attr in other_inputs.items():
                if not class_name == 'Particles':
                    self.__dict__[class_name.lower()].__dict__.update(class_attr)

        self.io.preprocessing = True
        self.io.setup()
        self.parameters.job_id = self.io.job_id
        # save some general info
        self.parameters.potential_type = self.potential.type
        self.parameters.cutoff_radius = self.potential.rc
        self.parameters.magnetized = self.integrator.magnetized
        self.parameters.integrator = self.integrator.type
        self.parameters.thermostat = self.thermostat.type

        self.parameters.setup(self.species)

        t0 = self.timer.current()
        self.timer.start()
        self.potential.setup(self.parameters)
        time_pot = self.timer.current()

        self.thermostat.setup(self.parameters)
        self.integrator.setup(self.parameters, self.thermostat, self.potential)
        self.timer.start()
        self.particles.setup(self.parameters, self.species)
        time_ptcls = self.timer.current()

        # For restart and backups.
        self.io.save_pickle(self)
        self.io.simulation_summary(self)
        time_end = self.timer.current()
        self.io.time_stamp("Potential Initialization", time_end - t0)
        self.io.time_stamp("Particles Initialization", time_ptcls - time_pot)
        self.io.time_stamp("Total Simulation Initialization", time_end - t0)

        self.kappa = self.potential.matrix[1, 0, 0] if self.potential.type == "Yukawa" else 0.0

    def green_function_timer(self):

        self.timer.start()
        self.potential.pppm_setup(self.parameters)
        return self.timer.stop()

    def run(self, loops=None, estimate=False):

        if loops:
            self.loops = loops

        self.estimate = estimate

        if self.potential.pppm_on:
            self.make_pppm_approximation_plots()
            green_time = self.green_function_timer()
            print('\n\n----------------- Force Calculation Times -----------------------\n')
            self.print_time_report("GF", green_time, 0)
        else:
            total_force_error, rcuts = self.analytical_approx_pp()
            print('\n\n----------------- Force Calculation Times -----------------------\n')

        self.time_acceleration()

        if self.estimate:
            self.estimate_best_parameters()

    def estimate_best_parameters(self):
        """Estimate the best number of mesh points and cutoff radius."""

        print('\n\n----------------- Timing Study -----------------------')

        max_cells = int(0.5 * self.parameters.box_lengths.min() / self.parameters.aws)
        if max_cells != len(self.pp_cells):
            self.pp_cells = np.arange(3, max_cells, dtype=int)

        pm_times = np.zeros(len(self.pm_meshes))
        pm_errs = np.zeros(len(self.pm_meshes))

        pp_times = np.zeros((len(self.pm_meshes), len(self.pp_cells)))
        pp_errs = np.zeros((len(self.pm_meshes), len(self.pp_cells)))

        pm_xlabels = []
        pp_xlabels = []

        self.force_error_map = np.zeros((len(self.pm_meshes), len(self.pp_cells)))

        # Average the PM time
        for i, m in enumerate(self.pm_meshes):

            self.potential.pppm_mesh = m * np.ones(3, dtype=int)
            self.potential.pppm_alpha_ewald = 0.25 * m / self.parameters.box_lengths.min()
            green_time = self.green_function_timer()
            pm_errs[i] = self.parameters.pppm_pm_err
            print('\n\nMesh = {} x {} x {} : '.format(*self.potential.pppm_mesh))
            print('alpha = {:1.4e} / a_ws = {:1.4e} '.format(self.potential.pppm_alpha_ewald * self.parameters.aws,
                                                             self.potential.pppm_alpha_ewald))
            print('PM Err = {:1.4e}  '.format(self.parameters.pppm_pm_err), end='')

            self.print_time_report("GF", green_time, 0)
            pm_xlabels.append("{}x{}x{}".format(*self.potential.pppm_mesh))
            for it in range(3):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_times[i] += self.timer.stop() / 3.0

            for j, c in enumerate(self.pp_cells):
                self.potential.rc = self.parameters.box_lengths.min() / c
                kappa_over_alpha = - 0.25 * (self.kappa / self.potential.pppm_alpha_ewald) ** 2
                alpha_times_rcut = - (self.potential.pppm_alpha_ewald * self.potential.rc) ** 2
                self.potential.pppm_pp_err = 2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(
                    self.potential.rc)
                self.potential.pppm_pp_err *= np.sqrt(self.parameters.total_num_ptcls) * self.parameters.aws ** 2 \
                                              / np.sqrt(self.parameters.box_volume)

                pp_errs[i, j] = self.potential.pppm_pp_err
                self.force_error_map[i, j] = np.sqrt(self.potential.pppm_pp_err ** 2
                                                     + self.parameters.pppm_pm_err ** 2)

                if j == 0:
                    pp_xlabels.append("{:1.2f}".format(self.potential.rc / self.parameters.aws))

                for it in range(3):
                    self.timer.start()
                    self.potential.update_linked_list(self.particles)
                    pp_times[i, j] += self.timer.stop() / 3.0

        self.lagrangian = np.empty((len(self.pm_meshes), len(self.pp_cells)))
        for i in range(len(self.pm_meshes)):
            for j in range(len(self.pp_cells)):
                self.lagrangian[i, j] = abs(pp_errs[i, j] ** 2 * pp_times[i, j] - pm_errs[i] ** 2 * pm_times[i])

        best = np.unravel_index(self.lagrangian.argmin(), self.lagrangian.shape)
        self.best_mesh = self.pm_meshes[best[0]]
        self.best_cells = self.pp_cells[best[1]]

        # set the best parameter
        self.potential.pppm_mesh = self.best_mesh * np.ones(3, dtype=int)
        self.potential.rc = self.parameters.box_lengths.min() / self.best_cells
        self.potential.pppm_alpha_ewald = 0.25 * self.best_mesh / self.parameters.box_lengths.min()
        self.potential.pppm_setup(self.parameters)

        # print report
        self.io.timing_study(self)
        # time prediction
        predicted_times = pp_times[best] + pm_times[best[0]]
        # Print estimate of run times
        self.io.time_stamp('Equilibration', predicted_times * self.parameters.equilibration_steps)
        self.io.time_stamp('Production', predicted_times * self.parameters.production_steps)
        self.io.time_stamp('Total Run',
                                predicted_times * (self.parameters.equilibration_steps
                                                    + self.parameters.production_steps) )

    def make_lagrangian_plot(self):

        c_mesh, m_mesh = np.meshgrid(self.self.pp_cells, self.self.pm_meshes)
        fig = plt.figure()
        ax = fig.add_subplot(111)  # projection='3d')
        # CS = ax.plot_surface(m_mesh, c_mesh, self.lagrangian, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        CS = ax.contourf(m_mesh,
                         c_mesh,
                         self.lagrangian,
                         norm=LogNorm(vmin=self.lagrangian.min(), vmax=self.lagrangian.max()))
        CS2 = ax.contour(CS, colors='w')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        fig.colorbar(CS)
        ax.scatter(self.best_mesh, self.best_cells, s=200, c='k')
        ax.set_xlabel('Mesh size')
        ax.set_ylabel(r'No. Cells = $1/r_c$')
        ax.set_title('2D self.lagrangian')
        fig.savefig(os.path.join(self.io.preprocessing_dir, '2D_Lagrangian.png'))
        fig.show()

    def make_force_error_map_plot(self):
        c_mesh, m_mesh = np.meshgrid(self.self.pp_cells, self.self.pm_meshes)
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        if self.force_error_map.min() == 0.0:
            minv = 1e-120
        else:
            minv = self.force_error_map.min()
        CS = ax.contourf(m_mesh,
                         c_mesh,
                         self.force_error_map,
                         norm=LogNorm(vmin=minv, vmax=self.force_error_map.max()))
        CS2 = ax.contour(CS, colors='w')
        ax.scatter(self.best_mesh, self.best_cells, s=200, c='k')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        fig.colorbar(CS)
        ax.set_xlabel('Mesh size')
        ax.set_ylabel(r'No. Cells = $1/r_c$')
        ax.set_title('Force Error')
        fig.savefig(os.path.join(self.io.preprocessing_dir, 'ForceMap.png'))
        fig.show()

    def time_acceleration(self):

        pp_acc_time = np.zeros(self.loops)
        pm_acc_time = np.zeros(self.loops)
        for i in range(self.loops):
            self.timer.start()
            self.potential.update_linked_list(self.particles)
            pp_acc_time[i] = self.timer.stop()
            self.timer.start()
            self.potential.update_pm(self.particles)
            pm_acc_time[i] = self.timer.stop()

        # Calculate the mean excluding the first value because that time include numba compilation time
        pp_mean_time = np.mean(pp_acc_time[1:])
        pm_mean_time = np.mean(pm_acc_time[1:])
        self.print_time_report("PP", pp_mean_time, self.loops)
        if self.potential.pppm_on:
            self.print_time_report("PM", pm_mean_time, self.loops)

        print('\n\n----------------- Estimated Simulation Times ---------------------\n')
        # Print estimate of run times
        eq_time = (pp_mean_time + pm_mean_time) * self.parameters.equilibration_steps
        self.io.time_stamp('Equilibration', eq_time)

        prod_time = (pp_mean_time + pm_mean_time) * self.parameters.production_steps
        self.io.time_stamp('Production', prod_time)

        tot_time = eq_time + prod_time
        self.io.time_stamp('Total Run', tot_time)

    def make_pppm_approximation_plots(self):
        chosen_alpha = self.potential.pppm_alpha_ewald * self.parameters.aws
        chosen_rcut = self.potential.rc / self.parameters.aws
        # Calculate Force error from analytic approximation given in Dharuman et al. J Chem Phys 2017
        total_force_error, pp_force_error, pm_force_error, rcuts, alphas = self.analytical_approx_pppm()

        # Color Map
        self.make_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

        # Line Plot
        self.make_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

    def make_line_plot(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot selected values of the total force error approximation.

        Parameters
        ----------
        rcuts: array
            Cut off distances.
        alphas: array
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: ndarray
            Force error matrix.

        parameters: class
            Simulation's parameters.

        """
        # Plot the results
        fig_path = self.io.preprocessing_dir

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 7))
        ax[0].plot(rcuts, total_force_error[30, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[30]))
        ax[0].plot(rcuts, total_force_error[40, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[40]))
        ax[0].plot(rcuts, total_force_error[50, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[50]))
        ax[0].plot(rcuts, total_force_error[60, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[60]))
        ax[0].plot(rcuts, total_force_error[70, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[70]))
        ax[0].set_ylabel(r'$\Delta F^{approx}_{tot}$')
        ax[0].set_xlabel(r'$r_c/a_{ws}$')
        ax[0].set_yscale('log')
        ax[0].axvline(chosen_rcut, ls='--', c='k')
        ax[0].axhline(self.parameters.force_error, ls='--', c='k')
        if rcuts[-1] * self.parameters.aws > 0.5 * self.parameters.box_lengths.min():
            ax[0].axvline(0.5 * self.parameters.box_lengths.min() / self.parameters.aws, c='r', label=r'$L/2$')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc='best')

        ax[1].plot(alphas, total_force_error[:, 30], label=r'$r_c = {:2.2f}'.format(rcuts[30]) + ' a_{ws}$')
        ax[1].plot(alphas, total_force_error[:, 40], label=r'$r_c = {:2.2f}'.format(rcuts[40]) + ' a_{ws}$')
        ax[1].plot(alphas, total_force_error[:, 50], label=r'$r_c = {:2.2f}'.format(rcuts[50]) + ' a_{ws}$')
        ax[1].plot(alphas, total_force_error[:, 60], label=r'$r_c = {:2.2f}'.format(rcuts[60]) + ' a_{ws}$')
        ax[1].plot(alphas, total_force_error[:, 70], label=r'$r_c = {:2.2f}'.format(rcuts[70]) + ' a_{ws}$')
        ax[1].set_xlabel(r'$\alpha \; a_{ws}$')
        ax[1].set_yscale('log')
        ax[1].axhline(self.parameters.force_error, ls='--', c='k')
        ax[1].axvline(chosen_alpha, ls='--', c='k')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc='best')
        fig.suptitle(
            r'Approximate Total Force error  $N = {}, \quad M = {}, \quad \kappa = {:.2f}$'.format(
                self.parameters.total_num_ptcls,
                self.potential.pppm_mesh[0],
                self.kappa * self.parameters.aws))
        fig.savefig(os.path.join(fig_path, 'ForceError_LinePlot_' + self.io.job_id + '.png'))
        fig.show()

    def make_color_map(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot a color map of the total force error approximation.

        Parameters
        ----------
        rcuts: array
            Cut off distances.

        alphas: array
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: ndarray
            Force error matrix.
        """
        # Plot the results
        fig_path = self.io.preprocessing_dir

        r_mesh, a_mesh = np.meshgrid(rcuts, alphas)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        if total_force_error.min() == 0.0:
            minv = 1e-120
        else:
            minv = total_force_error.min()
        total_force_error[ total_force_error == 0.0 ] = minv
        CS = ax.contourf(a_mesh, r_mesh, total_force_error, norm=LogNorm(vmin=minv, vmax=total_force_error.max()))
        CS2 = ax.contour(CS, colors='w')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        ax.scatter(chosen_alpha, chosen_rcut, s=200, c='k')
        if rcuts[-1] * self.parameters.aws > 0.5 * self.parameters.box_lengths.min():
            ax.axhline(0.5 * self.parameters.box_lengths.min() / self.parameters.aws, c='r', label=r'$L/2$')
        # ax.tick_parameters(labelsize=fsz)
        ax.set_xlabel(r'$\alpha \;a_{ws}$')
        ax.set_ylabel(r'$r_c/a_{ws}$')
        ax.set_title(
            r'$\Delta F^{approx}_{tot}(r_c,\alpha)$' + r'  for  $N = {}, \quad M = {}, \quad \kappa = {:1.2f}$'.format(
                self.parameters.total_num_ptcls, self.potential.pppm_mesh[0], self.kappa * self.parameters.aws))
        fig.colorbar(CS)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, 'ForceError_ClrMap_' + self.io.job_id + '.png'))
        fig.show()

    @staticmethod
    def print_time_report(str_id, t, loops):
        """Print times estimates of simulation."""
        if str_id == "GF":
            print("Optimal Green's Function Time = {:1.3f} sec \n".format(t))
        elif str_id == "PP":
            print('Average time of PP acceleration calculation over {} loops: {:1.3f} msec \n'.format(loops, t * 1e3))
        elif str_id == "PM":
            print('Average time of PM acceleration calculation over {} loops: {:1.3f} msec \n'.format(loops, t * 1e3))
    
    def analytical_approx_pp(self):
        """Calculate PP force error."""

        r_min = self.potential.rc * 0.5
        r_max = self.potential.rc * 1.5

        rcuts = np.linspace(r_min, r_max, 101) / self.parameters.aws

        # Calculate the analytic PP error and the total force error
        pp_force_error = np.sqrt(2.0 * np.pi * self.kappa) * np.exp(- rcuts * self.kappa)
        pp_force_error *= np.sqrt(self.parameters.total_num_ptcls *
                                  self.parameters.aws ** 3 / self.parameters.box_volume)

        return pp_force_error, rcuts

    def analytical_approx_pppm(self):
        """Calculate the total force error as given in Dharuman et al. J Chem Phys 146 024112 (2017)."""

        p = self.potential.pppm_cao
        L = self.parameters.box_lengths[0] / self.parameters.aws
        h = L / self.potential.pppm_mesh[0]

        a_min = self.potential.pppm_alpha_ewald * 0.5
        a_max = self.potential.pppm_alpha_ewald * 1.5

        r_min = self.potential.rc * 0.5
        r_max = self.potential.rc * 1.5

        alphas = self.parameters.aws * np.linspace(a_min, a_max, 101)
        rcuts = np.linspace(r_min, r_max, 101) / self.parameters.aws

        pm_force_error = np.zeros(len(alphas))
        pp_force_error = np.zeros((len(alphas), len(rcuts)))
        total_force_error = np.zeros((len(alphas), len(rcuts)))

        # Coefficient from Deserno and Holm J Chem Phys 109 7694 (1998)
        if p == 1:
            Cmp = np.array([2 / 3])
        elif p == 2:
            Cmp = np.array([2 / 45, 8 / 189])
        elif p == 3:
            Cmp = np.array([4 / 495, 2 / 225, 8 / 1485])
        elif p == 4:
            Cmp = np.array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
        elif p == 5:
            Cmp = np.array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
        elif p == 6:
            Cmp = np.array([2764 / 638512875, 16 / 467775, 7234 / 119282625, 1403744 / 25196700375,
                            1396888 / 40521009375, 2485856 / 152506344375])
        elif p == 7:
            Cmp = np.array([8 / 18243225, 7234 / 1550674125, 701872 / 65511420975, 2793776 / 225759909375,
                            1242928 / 132172165125, 1890912728 / 352985880121875, 21053792 / 8533724574375])

        kappa = self.kappa * self.parameters.aws

        for ia, alpha in enumerate(alphas):
            somma = 0.0
            for m in np.arange(p):
                expp = 2 * (m + p)
                somma += Cmp[m] * (2 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.) ** expp
            # eq.(36) in Dharuman J Chem Phys 146 024112 (2017)
            pm_force_error[ia] = np.sqrt(3.0 * somma) / (2.0 * np.pi)
        # eq.(35)
        pm_force_error *= np.sqrt(self.parameters.total_num_ptcls *
                                  self.parameters.aws ** 3 / self.parameters.box_volume)
        # Calculate the analytic PP error and the total force error
        if self.potential.type == "QSP":
            for (ir, rc) in enumerate(rcuts):
                pp_force_error[:, ir] = np.sqrt(2.0 * np.pi * kappa) * np.exp(- rc * kappa)
                pp_force_error[:, ir] *= np.sqrt(self.parameters.total_num_ptcls
                                                 * self.parameters.aws ** 3 / self.parameters.box_volume)
                for (ia, alfa) in enumerate(alphas):
                    # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
                    total_force_error[ia, ir] = np.sqrt(pm_force_error[ia] ** 2 + pp_force_error[ia, ir] ** 2)
        else:
            for (ir, rc) in enumerate(rcuts):
                for (ia, alfa) in enumerate(alphas):
                    # eq.(30) from Dharuman J Chem Phys 146 024112 (2017)
                    pp_force_error[ia, ir] = 2.0 * np.exp(-(0.5 * kappa / alfa) ** 2
                                                          - alfa ** 2 * rc ** 2) / np.sqrt(rc)
                    pp_force_error[ia, ir] *= np.sqrt(self.parameters.total_num_ptcls *
                                                      self.parameters.aws ** 3 / self.parameters.box_volume)
                    # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
                    total_force_error[ia, ir] = np.sqrt(pm_force_error[ia] ** 2 + pp_force_error[ia, ir] ** 2)

        return total_force_error, pp_force_error, pm_force_error, rcuts, alphas

    @staticmethod
    def make_fit_plot(pp_xdata, pm_xdata, pp_times, pm_times, pp_opt, pm_opt, pp_xlabels, pm_xlabels, fig_path):
        """
        Make a dual plot of the fitted functions.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].plot(pm_xdata, pm_times.mean(axis=-1), 'o', label='Measured times')
        ax[0].plot(pm_xdata, quadratic(pm_xdata, *pm_opt), '--r', label="Fit $f(x) = a + b x + c x^2$")
        ax[1].plot(pp_xdata, pp_times.mean(axis=-1), 'o', label='Measured times')
        ax[1].plot(pp_xdata, linear(pp_xdata, *pp_opt), '--r', label="Fit $f(x) = a x$")

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')

        ax[1].set_xscale('log')
        ax[1].set_yscale('log')

        ax[0].legend()
        ax[1].legend()

        ax[0].set_xticks(pm_xdata)
        ax[0].set_xticklabels(pm_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[1].set_xticks(pp_xdata[0:-1:3])
        ax[1].set_xticklabels(pp_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[0].set_title("PM calculation")
        ax[1].set_title("PP calculation")

        ax[0].set_xlabel('Mesh sizes')
        ax[1].set_xlabel(r'$r_c / a_{ws}$')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, 'Timing_Fit.png'))
        fig.show()


@njit
def Gk(x, alpha, kappa):
    """
    Green's function of Coulomb/Yukawa potential.
    """
    return 4.0 * np.pi * np.exp(-(x ** 2 + kappa ** 2) / (2 * alpha) ** 2) / (kappa ** 2 + x ** 2)


@njit
def betamp(m, p, alpha, kappa):
    """
    Calculate :math:`\beta(m)` of eq.(37) in Dharuman et al. J Chem Phys 146 024112 (2017)
    """
    xa = np.linspace(0.0001, 500, 5000)
    return np.trapz(Gk(xa, alpha, kappa) * Gk(xa, alpha, kappa) * xa ** (2 * (m + p + 2)), x=xa)


@njit
def analytical_approx_pppm_single(kappa, rc, p, h, alpha):
    """
    Calculate the total force error for a given value of ``rc`` and ``alpha``. See similar function above.
    """
    # Coefficient from Deserno and Holm J Chem Phys 109 7694 (1998)
    if p == 1:
        Cmp = np.array([2 / 3])
    elif p == 2:
        Cmp = np.array([2 / 45, 8 / 189])
    elif p == 3:
        Cmp = np.array([4 / 495, 2 / 225, 8 / 1485])
    elif p == 4:
        Cmp = np.array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
    elif p == 5:
        Cmp = np.array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
    elif p == 6:
        Cmp = np.array([2764 / 638512875, 16 / 467775, 7234 / 119282625, 1403744 / 25196700375,
                        1396888 / 40521009375, 2485856 / 152506344375])
    elif p == 7:
        Cmp = np.array([8 / 18243225, 7234 / 1550674125, 701872 / 65511420975, 2793776 / 225759909375,
                        1242928 / 132172165125, 1890912728 / 352985880121875, 21053792 / 8533724574375])

    somma = 0.0
    for m in np.arange(p):
        expp = 2 * (m + p)
        somma += Cmp[m] * (2 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.) ** expp
    # eq.(36) in Dharuman J Chem Phys 146 024112 (2017)
    pm_force_error = np.sqrt(3.0 * somma) / (2.0 * np.pi)

    # eq.(30) from Dharuman J Chem Phys 146 024112 (2017)
    pp_force_error = 2.0 * np.exp(-(0.5 * kappa / alpha) ** 2 - alpha ** 2 * rc ** 2) / np.sqrt(rc)
    # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
    Tot_DeltaF = np.sqrt(pm_force_error ** 2 + pp_force_error ** 2)

    return Tot_DeltaF, pp_force_error, pm_force_error


def quadratic(x, a, b, c):
    """
    Quadratic function for fitting.

    Parameters
    ----------
    x : array
        Values at which to calculate the function.

    a: float
        Intercept.

    b: float
        Coefficient of linear term.

    c: float
        Coefficient of quadratic term.

    Returns
    -------
    quadratic formula
    """
    return a + b * x + c * x * x


def linear(x, a):
    """
    Linear function for fitting.

    Parameters
    ----------
    x : array
        Values at which to calculate the function.

    a: float
        Coefficient of linear term.

    Returns
    -------
    linear formula
    """
    return a * x

