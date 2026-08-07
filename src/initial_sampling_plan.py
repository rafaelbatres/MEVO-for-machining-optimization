#generation of the set of samples and their responses (X,Y)
def initial_sampling_plan(ndim, n_samples, upper_bound, lower_bound, obj_func): 
    #Import required libraries
    import pandas as pd
    from skopt.space import Space
    from skopt.sampler import Lhs

    space = Space([(0.0, 1.0) for _ in range(ndim)])
    dimensions = space.dimensions

    lhs = Lhs(criterion="maximin", iterations=100000)
    Xsamples_org = lhs.generate(space.dimensions, n_samples)

    # Scale and shift samples
    for sample in Xsamples_org:
        for i in range(ndim):
            sample[i] *= (upper_bound[i] - lower_bound[i])
            sample[i] += lower_bound[i]

    Xsamples = Xsamples_org


    # Initialize sampling file data structures
    data_files = [[] for _ in range(ndim)]
    data_file_y = []
    yactual = []

    for x in Xsamples:
        for i in range(ndim):
            data_files[i].append(x[i])
        y_calc = obj_func(*x)
        data_file_y.append(y_calc)
        yactual.append(y_calc)

    # Create a sampling plan data frame
    df = pd.DataFrame()
    for i in range(ndim):
        df[f'x{i+1}'] = pd.Series(data_files[i])
    df['y'] = pd.Series(data_file_y)

    # Save sampling plan to a CSV file
    df.to_csv('initial_points.csv', index=False)
