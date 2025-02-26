#!/usr/bin/env python
# coding: utf-8

import os 
from multiprocessing import Process
from numpy.random import Generator, Philox, SeedSequence

from sarkas.processes import PostProcess, Simulation
from sarkas.tools.observables import RadialDistributionFunction, Thermodynamics


def simulate(seed: int, run_no: int, input_file_name: str):
    """
    Run an MD simulation.

    Parameters:
        seed (int): The random seed for particle initialization.
        n_eq (int): The number of equilibration steps to further randomize the runs.
        run_no (int): The run number.
        input_file_name (str): The name of the input file.

    Returns:
        None
    """
    args = {
        "Parameters": {
            "rand_seed": seed,          # Random seed for particle initialization
        },
        "IO": {
            "verbose": False,   # No need to output the progress bar.
            # The directory where the simulations will be saved. 
            # SarkasSimulations is the default directory where the simulations are saved.
            "md_simulations_dir": 'SarkasSimulations', 
            "job_dir": f"run{run_no}",
            # Note that we don't want to overwrite each simulation
            # So we save each simulation in its own folder by passing
            # a dictionary of dictionary with folder's name
        },
    }
    sim = Simulation(input_file_name)
    sim.setup(read_yaml=True, other_inputs=args)
    sim.run()


def postprocessing(run_no, input_file_name):
    """
    Perform post-processing tasks on the simulation results.

    Args:
        run_no (int): The run number of the simulation.
        input_file_name (str): The name of the input file.

    Returns:
        None
    """

    # Define the arguments for post-processing
    args = {
        # No need to pass Parameters again, because Sarkas will read the pickle file and extract the parameters.
        "IO": {
            "verbose": False,
            # The directory where the simulations will be saved. 
            # SarkasSimulations is the default directory where the simulations are saved.
            "md_simulations_dir": 'SarkasSimulations',  
            "job_dir": f"run{run_no}",
            # Note that we don't want to overwrite each simulation
            # So we save each simulation in its own folder by passing
            # a dictionary of dictionary with folder's name
        },
    }

    # Create a PostProcess object
    postproc = PostProcess(input_file_name)
    postproc.setup(read_yaml=True, other_inputs=args)

    therm = Thermodynamics()
    therm.setup(postproc.parameters)
    therm.compute() 

    # Radial Distribution Functions
    rdf = RadialDistributionFunction()
    rdf.setup(postproc.parameters)
    rdf.compute()

    # Make a plot of the RDF
    ax = rdf.plot(
        scaling=rdf.a_ws,  # Scale by the Wigner-Seitz radius
        y = [("Sr-Sr RDF", "Mean")], # Column name
        xlabel = r'$r/a_{\rm ws}$', 
        ylabel = r'$g(r)$'
    )
    ax.legend(["Sr-Sr RDF"])  # Remake the legend.
    # The plot will be saved automatically.

if __name__ == "__main__":

    input_file = os.path.join('input_files', 'ucp_N10k.yaml')
    # Define a random number generator
    sg = SeedSequence(654654619)

    total_number_of_runs = 5
    # Spawn ten random number generators
    rng_s= [Generator(Philox(s)) for s in sg.spawn(total_number_of_runs)]
    # Read this link for more information on why we use Philox
    # https://numpy.org/doc/stable/reference/random/parallel.html


    # Create a list to store the processes
    processes = []

    # Start 10 runs in parallel
    for i in range(0, total_number_of_runs):
        # Generate a random seed and number of equilibration steps using the random number generator
        seed = rng_s[i].integers(987650,341865651)        
        # Create a new process that calls the 'simulate' function with the generated seed, n_eq, i, and input_file as arguments
        p0 = Process(target=simulate, args=(seed, i, input_file,))
        
        # Add the process to the list
        processes.append(p0)
        
        # Start the process
        p0.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

        
    # Post Processing
    processes = []

    # Iterate over the runs
    for i in range(0, total_number_of_runs):
        # Create a new process that calls the 'postprocessing' function with the current index 'i' and 'input_file' as arguments
        p0 = Process(target=postprocessing, args=(i, input_file,))
        
        # Add the process to the list
        processes.append(p0)
        
        # Start the process
        p0.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()
