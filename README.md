# DLProject

The goal of this project is to test different deep learning models
against different datasets. 

#### How to run 

To run the benchmarks run the solution finder with all data 
on the data_files folder
> python3 solution_finder.py
 
 
 ### Check results
 
 If all goes well the results of the run will be stored on a
 log file in the log_files directory
 
 The log file have the following structure:
 
 Seed,Algorithm,dataset,time,train_f1,val_f1,acc,val_acc,bin_acc,val_bin_acc,train_loss,val_loss


### Understanding the code

The solution_finder.py yields most information on what is being run  
The ML_algoritms.py as the networks with the different architectures
The utils.py contains most data operations required to feed the data to the model
and some other general utilities functions

### Notebooks 
The notebooks are used to perform some analysis prior to 
adding the models to the solution finder