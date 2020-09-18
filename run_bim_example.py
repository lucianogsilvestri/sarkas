from sarkas.processes import PreProcess, Simulation, PostProcess
from numpy.random import Generator, PCG64

input_file_name = 'sarkas/examples/coulomb_bim_mks.yaml'

rg = Generator(PCG64(12345))

# preproc = PreProcess(input_file_name)
# preproc.setup(read_yaml=True)
# preproc.run(loops=5)

# sim = Simulation(input_file_name)
# sim.setup(read_yaml=True)
# sim.run()
    #
postproc = PostProcess(input_file_name)
postproc.setup(read_yaml=True)
#
# postproc.therm.setup(postproc.parameters, postproc.species)
# postproc.therm.temp_energy_plot(postproc, phase='equilibration', show=False)
# postproc.therm.temp_energy_plot(postproc, phase='production', show=False)
# postproc.therm.plot('Temperature', show=False)
#
# postproc.rdf.setup(postproc.parameters)
# postproc.rdf.save()
# postproc.rdf.plot(show=False)
# #
# postproc.hc.setup(postproc.parameters, 'equilibration')
# postproc.hc.parse()
# postproc.hc.plot(show=False)
# #
# postproc.hc.setup(postproc.parameters, 'production')
# postproc.hc.parse()
# postproc.hc.plot(show=False)
# #
# postproc.vm.setup(postproc.parameters, 'equilibration')
# postproc.vm.parse()
# postproc.vm.plot_ratios(show=False)
# #
# postproc.vm.setup(postproc.parameters, 'production')
# postproc.vm.parse()
# postproc.vm.plot_ratios(show=False)
#
postproc.dsf.setup(postproc.parameters)
postproc.dsf.parse()
postproc.dsf.plot(show=False)
#
postproc.ccf.setup(postproc.parameters)
postproc.ccf.parse()
postproc.ccf.plot(show=False)
#
postproc.ssf.setup(postproc.parameters)
postproc.ssf.parse()
postproc.ssf.plot(show=False)