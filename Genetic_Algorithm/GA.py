

from Genetic_Algorithm import GA_functions
import numpy as np
import matplotlib.pyplot as plt


def algorithm(array_train, array_val, classif):

    # Loading the necessary variables for applying GA

    pop_size = 50 # Population size
    num_parents_mating = (int)(pop_size*0.5) # Number of parents inside the mating pool.
    num_mutations = 3 # Number of elements to mutate.
    num_generations = 10 # Number of generations
    classifier = classif

    num_feature_elements_train = array_train.shape[1]-1


    # Applying Genetic Algorithm (GA)

    # Defining the population shape.
    pop_shape = (pop_size, num_feature_elements_train)

    # Creating the initial population.
    new_population = np.random.randint(low=0, high=2, size=pop_shape)
    print(new_population.shape)

    best_outputs = []
    for generation in range(num_generations):
        print("Generation : ", (generation+1))
        # Measuring the fitness of each chromosome in the population.
        fitness = GA_functions.cal_pop_fitness(new_population, array_train, array_val, classifier)

        best_outputs.append(np.max(fitness))
        # The best result in the current iteration.
        print("Best result : ", best_outputs[-1])

        # Selecting the best parents in the population for mating.
        parents = GA_functions.select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        crossed_offsprings = GA_functions.crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements_train))

        # Adding some variations to the offspring using mutation.
        mutated_offsprings = GA_functions.mutation(crossed_offsprings, num_mutations)

        # Creating the new population based on the parents and offspring.
        new_population[ :parents.shape[0], :] = parents
        new_population[parents.shape[0]: , :] = mutated_offsprings


    # Getting the best solution after iterating finishing all generations.

    best_match_idx = np.where(best_outputs == np.max(best_outputs))[0]
    best_match_idx = best_match_idx[0]

    best_acc = (best_outputs[best_match_idx])*100.0
    best_solution = new_population[best_match_idx, :]
    best_solution_indices = np.where(best_solution == 1)[0]
    best_solution_num_elements = best_solution_indices.shape[0]

    # Printing the required statistics

    print("The accuracy of the best candidate solution is {:.4f}".format(best_acc))
    print("Selected feature indices by GA : ", best_solution_indices)
    print("Number of selected features by GA : ", best_solution_num_elements)

    # Plotting the 'Accuracy' vs 'Generation' curve

    plt.plot(range(num_generations), best_outputs,'b')
    plt.xlabel('Generations')
    plt.ylabel("Accuracy")