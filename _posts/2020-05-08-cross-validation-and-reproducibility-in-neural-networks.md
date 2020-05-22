# Cross Validation and Reproducibility in Neural Network Training
Neural networks have a high tendency to overfit on training data, especially when the examples are few and the network has a large capacity. There is even a famous quote which says, A popular deep learning adage is that "If your neural network is not overfitting, then it is not learning enough. It just matters how much you want it to." But with this comes its own problem; the ability to generalize to unseen data. Cross validation is a great tool to mimic generalization accuracy, and it is especially important for experiments with very few training examples. Reproducibility, on the other hand, ensures that we can repeat our experiments several times, without much randomness in the results (as is the case in cross validation).  

Here's what we will cover:
1. TOC
{:toc}

## Introduction  
Cross-validation is a resampling technique that assesses how the results of a statistical analysis will generalize to an independent data set. Three commonly used types are; i) K-fold cross validation, ii) a variant called Stratified K-fold cross validation and iii) the leave-one-out cross validation.   
Given data samples $\{(x_1, y_1), (x_2, y_2), ... (x_n, y_n)\}$  where n is the total number of examples, $\textbf{x}_i$ is a d-dimensional vector or a tensor (as in images), and $y_i$ is the class or label of example, $i$

- The k-fold cross validation is the standard type. The training data is split into k different parts. k is an integer (usually between 5-10), and depends on the size of data). k < total number of examples.
- If k = total number of examples, then, the k-fold becomes leave-one-out cross validation, as only one example is placed in the validation set in each validation run.
- The stratified k-fold cross validation is suitable for instances where there is imbalance in the frequency of the classes. If we use a random sampling, as is the case in k-fold, some examples might not have enough contribution to some folds.

I think I have only talked about cross validation on a high level, and I assume you are already familiar with the concept in machine learning. You might need to check out other sources if you require more details.

## Why Reproducibility in Training?
Reproducibility ensures that we can recreate experiments. Deep learning models have been notoriously known to have many parameters and hyperparameters, with randomness in initializations and sampling. For reproducibility sake, we have to put things in order to validate our experiments. Two main places to ensure this are; i) hyperparameter initializations and ii) random seed settings

### Structuring Hyperparameters
Firstly, we will define our hyperparameters and other settings in a structured way, so that hyperparameter changes can only be applied only at one single point. We can use a dictionary for this purpose or the argparse library. The argparse library is more preferred as it is an argument parsing tool which helps to easily translate from hyperparameters in notebooks to arguments on the command line. For an in-depth tutorial on this check out this [argparse tutorial](https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05) by Jeff Hale on Medium.

The setup is simple and looks like this
```
>> from argparse import Namespace

>> args = Namespace(
      size = 448,

      # Model Hyperparameters
      learning_rate = 1e-4,
      batch_size = 8,
      num_epochs = 10,
      early_stopping_criteria=10,
      momentum=0.9,

      # CV parameter
      num_folds=5,

      seed=0,

      # Runtime hyper parameter
      cuda=True,
      )
```

Then you can easily call or modify the parameters, like a normal python dictionary. For example, to add a device argument
```
# Check CUDA
>> if not torch.cuda.is_available():
       args.cuda = False
    
>> # Add device for training
>> args.device = torch.device("cuda" if args.cuda else "cpu")
```

### Setting a global random seed
Our computers only generate pseudo-random numbers. This means that we can make them generate the same set on random numbers continuously if we set a starting seed. Recall that neural network training requires different libraries such as numpy, pytorch, pandas, cudnn etc interfacing. They also make use of random number generators which all require seeds. Also, many parts of the neural network model itself - such as weights, biases, dropout - require sampling.  
How can we account for these randomness in our experiments?  
The simple solution is to set a single seed and consistently apply this seed across all the libraries requiring it.  

```
def setup_seed(seed, cuda):
    # Creates global random seed across torch, cuda and numpy 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```
Then, we can easily call the function to set up the seeding across all libraries.
```
>> setup_seed(args.seed, args.cuda)
```

That's it!! Now that you understand reproducibility, let me walk you through a cross validation process in neural networks. I will be applying it to a classification task.

## Cross validation applied to neural network
Cross validation can be used to select the best hyperparameters for training a neural network. If the folds have good performance on their validation sets when a set of hyperparameters is applied, then that set of hyperparameters is believed will help generalize the model to unseen data. In the same vein, the Cross validation models can likewise be ensembled in several different ways for prediction;
- the weights of all the folds can be averaged to get a more robust model. This is popularly known as Polyak Averaging.
- the model of each fold can be saved to make predictions on an unseen data. Then, softmax predictions of examples of each fold can be averaged to predict the correct class. This is popularly known as model ensemble.  

I will be using the [Cassava Disease Challenge](https://www.kaggle.com/c/ammi-2020-convnets) on Kaggle as a running example to explain the underlying concept. The challenge entailed classifying pictures of cassava leaves into 1 of 4 disease categories or healthy.   

I will only highlight parts of the code that performs the cross-validation and will not try to show all codes here. For the full code, you can check the [cassava disease classification](https://www.kaggle.com/ogunlao/crossvalidation-for-cassava-disease-classification) Kaggle kernel I created for this tutorial.

### The (Stratified) K-fold cross validation
Every task has its own peculiarity and understanding the statistics of the data can go a long way in getting good results. For instance, the number of cassava leaf samples in each disease category differs by a large margin, creating a data imbalance. To ensure each class is involved in each fold, we can "bias" the sampling using the Stratified K-fold cross validation.  

Using 5-fold cross validation splits the data into 80% training and 20% validation (which is a popular choice).

```
def stratified_kfold(num_folds=5, images_df=None):
    st_kfold = StratifiedKFold(n_splits=num_folds, shuffle=args.shuffle_dataset, random_state=args.seed)

    fold = 0
    for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):
        train, val = images_df.iloc[train_index], images_df.iloc[val_index]

        train_dataset = CassavaDataset(df_data=train, transform=train_trans)
        valid_dataset = CassavaDataset(df_data=val,transform=val_trans)

        train_loader = DataLoader(dataset = train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=args.shuffle_dataset, 
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset = valid_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=args.shuffle_dataset, 
                                  num_workers=args.num_workers)

        dataloaders = {'train': train_loader, 'val': valid_loader}
        
        dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}
        print(dataset_sizes)

        print()
        print(f'Starting CV for Fold {fold}')
        model_ft, criterion, optimizer_ft, exp_lr_scheduler = prepare_model(pretrained=args.pretrained)
        model_ft = train_model(model_ft, 
                              criterion, 
                              optimizer_ft, 
                              exp_lr_scheduler,
                              dataloaders,
                              dataset_sizes,
                              num_epochs=args.num_epochs,)
        
        # Save model for the current fold to your output directory
        current_fold_full_path = args.save_dir +'/model_'+str(fold)+'_tar'
        torch.save(model_ft.state_dict(), current_fold_full_path)
        
        fold += 1
```

- A dataframe containing the name/location of the images and label is passed into the stratified_kfold function, with the number of folds
- The k-fold function splits the data into 5-folds and returns the indexes of the examples belonging to train and validation set in each fold.
- These indexes are then used to retrieve the examples to create train and validation datasets.
- The dataloaders then create batches of data for training and validation.  
Note that the definitions for train_model, Dataloaders and Datasets are not shown here, but are available in the [cassava disease classification](https://www.kaggle.com/ogunlao/crossvalidation-for-cassava-disease-classification) Kaggle kernel setup for this tutorial for reference.

###  The model initialization
One subtle problem that has not yet been considered thus far is how to initialize the model in each fold. Do we continue with the model used in a previous fold or we initialize a new model? Well, it is clear, we have to reinitialize to a new but similar model on every fold. There are 3 different initialization cases to be considered;  
1. Using a pretrained model with downloadable weights.
This is pretty straight forward. Just redownload the weights or reload the weight from the stored cache at the beginning of each fold.
2. Using a custom model built from scratch.
Just reinitialize your model on each fold. Since we have set a seed, we are more likely to have the same set of weights and biases during reinitialization. You can also initialize once, store the weights in a cache and retrieve the stored weights for each fold.
3. Using a base model architecture such as ResNet without pretrained weights.
Similar to 2 above. Just make sure you reinitialize your model on each fold.  
> In any case, do not use the trained model of a previous fold to initialize a new fold. The training outcome can be very*5 disastrous and deceptive!!

### Train on the entire dataset
Now that you have performed cross validation with your hyperparameters, you can then use the best hyperparameters (such as learning rate, number of epochs, batch size, optimizer, image size etc) to retrain the model on the entire training samples. This might bring about some significant improvement as the dataset can now see more examples in a single run.

## Conclusion
Cross validation is one main topic less talked about in deep learning due to the time it takes to perform the process. It can also be less favoured when there is a large training set (in the range of millions of examples). In cases where the sample size is not massive, we still require some classical ideas such as cross validation to generalize to unseen data. Reproducibility ensures that on each fold of cross validation, we can reinitialize our model to the original state.
