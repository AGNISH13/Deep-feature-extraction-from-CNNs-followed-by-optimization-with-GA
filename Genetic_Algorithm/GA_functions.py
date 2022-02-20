

import numpy as np

# Defining the necessary functions for GA


# Feature Reduction to eliminate the redundant features

def reduce_features(solution, array_train, array_val):
    solution = np.append(solution, [1])
    selected_elements_indices = np.where(solution == 1)[0] # Selecting elements whose genes are given a value of 1
    reduced_train_features = array_train[:, selected_elements_indices]
    reduced_val_features = array_val[:, selected_elements_indices]
    return reduced_train_features, reduced_val_features


# Calculating and returning accuracy of population

def classification_accuracy(labels, val_predictions):
    correct = np.where(labels == val_predictions)
    accuracy = correct[0].shape[0]/val_predictions.shape[0]
    return accuracy
    

# Calculating the population fitness using SVM, KNN or MLP classifiers

def cal_pop_fitness(pop, array_train, array_val, classifier):
    accuracies = np.zeros(pop.shape[0])
    idx = 0 # Counter variable for creating the array accuracies

    for curr_solution in pop: # Current solution is the chromosome and pop is the total set of chromosomes (population)
        reduced_train_features, reduced_val_features = reduce_features(curr_solution, array_train, array_val)
        X=reduced_train_features[:,:-1] # Taking all the features columns
        y=reduced_train_features[:,-1]  # Taking the labels column
        reduced_validation_features = reduced_val_features[:,:-1]
        validation_labels = reduced_val_features[:,-1]
            
        # Mentioning classifier

        if classifier == 'SVM':
            ## SVM CLASSIFIER ##
            from sklearn.svm import SVC
            SVM_classifier = SVC(kernel='rbf')
            SVM_classifier.fit(X, y)
            val_predictions = SVM_classifier.predict(reduced_validation_features)

        elif classifier == 'MLP':
            ## MLP CLASSIFIER ##
            from sklearn.neural_network import MLPClassifier
            MLP_classifier = MLPClassifier()
            MLP_classifier.fit(X, y)
            val_predictions = MLP_classifier.predict(reduced_validation_features)


        else:
            ## KNN CLASSIFIER ##
            from sklearn.neighbors import KNeighborsClassifier
            KNN_classifier = KNeighborsClassifier(n_neighbors=2)
            KNN_classifier.fit(X, y)
            val_predictions = KNN_classifier.predict(reduced_validation_features)

        accuracies[idx] = classification_accuracy(validation_labels, val_predictions)     
                
        idx = idx + 1
    return accuracies


# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_index = np.where(fitness == np.max(fitness))[0]
        max_fitness_index = max_fitness_index[0]
        parents[parent_num, :] = pop[max_fitness_index, :]
        fitness[max_fitness_index] = -99999999999
    return parents


# Applying One Point Crossover

def crossover(parents, offspring_size):
    crossed_offsprings = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_index = k%parents.shape[0]
        
        # Index of the second parent to mate.
        parent2_index = (k+1)%parents.shape[0]
        
        # The new offspring will have its first half of its genes taken from the first parent.
        crossed_offsprings[k, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        
        # The new offspring will have its second half of its genes taken from the second parent.
        crossed_offsprings[k, crossover_point:] = parents[parent2_index, crossover_point:]
    return crossed_offsprings


# Performing Mutation of randomly selected genes by flipping their values

def mutation(crossed_offsprings, num_mutations):
    mutation_index = np.random.randint(low=0, high=crossed_offsprings.shape[1], size=num_mutations)
    for index in range(crossed_offsprings.shape[0]):
        # The random value to be added to the gene.
        crossed_offsprings[index, mutation_index] = 1 - crossed_offsprings[index, mutation_index]
    return crossed_offsprings