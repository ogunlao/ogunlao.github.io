# Cross Validation and Reproducibility in Neural Network Training
Neural networks have a high tendency to overfit on data, especially when the examples and few and the network has a large capacity. There is even a famous quote which says, "If your neural network does not overfitting, then it is not learning enough". But with this comes it quirks; the ability to generalize to unseen data. To ensure this, cross validation is a great tool, especially when there are very few examples for training. Reproducibility, on the other hand, ensures that, if the model is run several times, the cross validation gives the same results.

Here's what we will cover:
1. TOC
{:toc}

## Introduction
Cross-validation is a resampling algorithm which is used to evaluate machine learning models on limited data sample. It is a validation technique that assesses how the results of a statistical analysis will generalize to an independent data set. There are common used types names; K-fold cross validation, a variant called Stratified K-fold cross validation and the leave-one-out cross validation.   
Given data samples $\{\textbf{x}_i, y_i\}^n_{i=1}$  where n is the total number of examples, $\textbf{x}_i$ is a d-dimensional vector or a tensor (as in images), and $y_i$ is the class or label of example, $\textbf{x}_i$

- The k-fold cross validation is the standard type where the data is split into k different parts where k is an integer (usually between 5-10, and depends on the data size) and k < total number of examples.
- If k = total number of examples, the the k-fold becomes leave-one-out cross validation, as only one example is used as validation set in each validation run.
- The stratified k-fold cross validation is suitable for instances where there is inbalance in the frequency of the classes. If we use a random sampling, as is the case in the case of k-fold, some examples might not have enough contribution to some folds.

I think I have only talked about cross validation on a high level, and i assume you are already familiar with the concept in machine learning. You might need to check out other sources if your require more details.

## Why Reproducibility in Training?
How would you feel if you run the same experiment multiple times and at those different times, you get different results, without being able to account . You might not know where you are going wrong, or what to improve upon. Reproducibility ensures that you can recreate experiments. Deep learning models has been notorious known to have many parameters and hyperparameters. For reproducibility sake, we have to follow a standard process to ensure things are in order.

### Structuring Hyperparameters
First we should define our hyperparameters and other setting in a structured way, so that hyperparameter changes can only be applied at only one single point. We can use a dictionary for this purpose or the argparse library. The argparge library is more prefered as it is an argument parsing tool which helps to easily translate from hyperparameters in notebooks to arguments on the command line. For an indepth tutorial on this check out this tutorial by Jeff Hale on [argparse](https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05)

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

### Setting the random seed correctly
Our computers only generate pseudo-random numbers. This means that we can make them generate the same set on random numbers continuously if we set a seed. The major challenge with neural network training is that, it combines different libraries such as numpy, pytorch, pandas, cudnn etc which all require seeds. Also recall that many parts of the neurl network model itself - such as weight intitialization, dropout - require randomness. We may not be able to account for all the randomness but we can atleast try.
For reproducibility, we can set a single seed as we did above in out hyperparameter settings and apply this seed across all the libraries.

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
The, we can easily call the function to setup the seeding
```
>> setup_seed(args.seed, args.cuda)
```

That's it!! Now that we are done with reproducibility, let me walk you through a cross validation process in neural networks. I will be applying it to a classification task.

## Cross validation applied to neural network.
Cross validation can be used to select the best hyperparameters for training a neural network. If the folds agree on best performance of the hyperparameters, then the set of hyperparameters can be considered for training the network. The Cross validation models can be ensembled in several different ways even after selecting the best parameters;
- the weights of all the folds can be averaged to get a more robust model.
- the softmax predictions of each fold can be averaged, before finally selecting the class with maximum probability for each validation example.
- the final predictions of each model can undergo a voting process, selecting the class with highest count for each fold.

I will be using the [Cassava Disease Challenge](https://www.kaggle.com/c/cassava-disease) on Kaggle as a runnung example to explain the underlying concept. The challenge entailed classifying pictures of cassava leaves into 1 of 4 disease categories.  

I will only highlight parts of the code that does cross-validation and will not try to show all codes here. For the full code, you can check it on github repo [cassava disease classification](https://ogunlao.github.io)  

### The (Stratified) K-fold cross validation.
Every task has its own peculiarity and understanding the statistics of the data can go a long way in getting good results. For example, the number of cassava diseases examples in each class differs by a large margin. This creates an imbalance of data samples. To ensure, each class can be involved in each fold, we can use the Stratified K-fold cross validation to create a "biased" sampling.  

The 5-fold stratified cross validation splits the data into 80% training and 20% validation (which is a popular choice).

```
def stratified_kfold(num_folds=args.num_folds, image_df):
    st_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)

    fold = 0
    for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):
        train, val = images_df.iloc[train_index], images_df.iloc[val_index]

        train_dataset = CassavaDataset(df_data=train, transform=train_trans)
        valid_dataset = CassavaDataset(df_data=val,transform=val_trans)

        train_loader = DataLoader(dataset = train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=4)
        valid_loader = DataLoader(dataset = valid_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  num_workers=4)

        dataloaders = {'train': train_loader, 'val': valid_loader}
        
        dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}
        print(dataset_sizes)

        print()
        print(f'Starting CV for Fold {fold}')
        # init_model = copy.deepcopy(model)
        model_ft = train_model(model_ft, 
                              criterion, 
                              optimizer_ft, 
                              exp_lr_scheduler, 
                              num_epochs=arg.num_epochs)

        fold += 1

print('Cross Validation Done ...')
```

- A dataframe containing the name/location of the images and label is passed into the stratified_kfold function, with the number of folds
- The k-fold function splits the data into the k-folds and returns the index of the example belonging to train and validation set in each fold.
- These indexes are used to retrieve the examples, to create a train and validation dataset.
- This in turn is passed to the dataloaders.
- The dataloaders creates batch of data for traing and validation.  
Note that the definitions for train_model, Dataloaders and Datasets are not shown, but are available on this [github link](https://ogunlao.github.io) for reference.

###  The model initialization
One idea is missing thus far and has not yet been considered, the model. The model used in each fold must be exactly the same for fair comparism. There are 3 different cases to be considered
1. Using pretrained model with downloadable weights.
This is pretty straight forward. Just redownload the weights or reload the weight from the stored cache for each fold.
2. Using a custom model built from scratch.
Just reinitialize your model on each fold. Since, we have set a seed, we are more likely to have the same weight. You can also intialize once, store the weights and retrieve the stored weights for each fold.
3. Using a base model architecture such as ResNet without pretrained weights
Similar to 2 above. Just make sure you reinitialize your model on each fold.
> In any case, do not use the trained model of a previous fold to intialize a new fold. This is very*5 disastrous!!

### Training on the entire dataset
Now that you have performed cross validation with your hyperparameters, you can also use the best hyperparameters (such as learning rate, number of epochs, batch size, optimizer, image size etc) to retrain the model on the whole dataset. This might bring about some significan improvement as the dataset can now see more examples in a single run.

## Conclusion
Cross validation is one main topics less talked about in deep learning due to the enormous amount of data. In cases where the sample size is not massive, we still require some classical ideas to get the best results. Reproducibility ensures that on each fold of cross validation, we can reinitialize our model to the original state.
