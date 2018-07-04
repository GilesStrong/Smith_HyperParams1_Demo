from __future__ import division
import timeit
import pandas
import numpy as np
import os

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.regularizers import *
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from Modules.Callbacks import *
from Modules.Misc_Functions import uncertRound
from Modules.Plotters import *
from Modules.AMS import amsScanQuick

pandas.options.mode.chained_assignment = None

def getPreProcPipes(normIn=False, pca=False):
    '''Constructs a SK-Learn pipeline to preprocess data'''
    stepsIn = []

    if not normIn and not pca:
        stepsIn.append(('ident', StandardScaler(with_mean=False, with_std=False))) #For compatability

    else:
        if pca:
            stepsIn.append(('pca', PCA(whiten=False)))
        if normIn:
            stepsIn.append(('normIn', StandardScaler()))

    inputPipe = Pipeline(stepsIn)
    return inputPipe

def getModel(version, nIn, compileArgs):
    '''Build Keras model with a few arguments
    NB: modelSwish requires swish activation function to be added to [keras location]/activations.py.
    def swish(x):
        return x*sigmoid(x)'''
    model = Sequential()

    if 'depth' in compileArgs:
        depth = compileArgs['depth']
    else:
        depth = 3
    if 'width' in compileArgs:
        width = compileArgs['width']
    else:
        width = 100
    if 'do' in compileArgs:
        do = compileArgs['do']
    else:
        do = False
    if 'bn' in compileArgs:
        bn = compileArgs['bn']
    else:
        bn = False
    if 'l2' in compileArgs:
        reg = l2(compileArgs['l2'])
    else:
        reg = None

    if "modelRelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('relu'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('relu'))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    elif "modelSelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='lecun_normal', kernel_regularizer=reg))
        model.add(Activation('selu'))
        if do: model.add(AlphaDropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg))
            model.add(Activation('selu'))
            if do: model.add(AlphaDropout(do))

    elif "modelSwish" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('swish'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('swish'))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))

    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        if 'amsgrad' not in compileArgs: compileArgs['amsgrad'] = False
        if 'beta_1' not in compileArgs: compileArgs['beta_1'] = 0.9
        optimiser = Adam(lr=compileArgs['lr'], beta_1=compileArgs['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compileArgs['amsgrad'])

    if compileArgs['optimizer'] == 'sgd':
        if 'momentum' not in compileArgs: compileArgs['momentum'] = 0.9
        if 'nesterov' not in compileArgs: compileArgs['nesterov'] = False
        optimiser = SGD(lr=compileArgs['lr'], momentum=compileArgs['momentum'], decay=0.0, nesterov=compileArgs['nesterov'])
    
    if 'metrics' not in compileArgs: compileArgs['metrics'] = None
    model.compile(loss=compileArgs['loss'], optimizer=optimiser, metrics=compileArgs['metrics'])
    return model

def trainClassifier(model, train, val, trainParams, 
                    patience=10, useEarlyStop=True, saveBest=True,
                    useCallbacks={}):
    '''Basic function to train model on provided data'''
    callbacks=[]
    if useEarlyStop:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')) #Setup early stopping

    if saveBest:
        callbacks.append(ModelCheckpoint("train_weights/best.h5", monitor='val_loss', verbose=0, #Save best performing network
                                     save_best_only=True, save_weights_only=True, mode='auto', period=1))

    if len(useCallbacks):
        nb = math.ceil(len(train['x'])/trainParams['batch_size'])

    if 'OneCycle' in useCallbacks:
        callbacks.append(OneCycle(**{'nb':nb, **useCallbacks['OneCycle']}))
    else:
        if 'LinearCLR' in useCallbacks:
            callbacks.append(LinearCLR(**{'nb':nb, **useCallbacks['LinearCLR']}))

        elif 'CosAnnealLR' in useCallbacks:
            callbacks.append(CosAnnealLR(**{'nb':nb, **useCallbacks['CosAnnealLR']}))

        if 'LinearCMom' in useCallbacks:
            callbacks.append(LinearCMom(**{'nb':nb, **useCallbacks['LinearCMom']}))

        elif 'CosAnnealMom' in useCallbacks:
            callbacks.append(CosAnnealMomentum(**{'nb':nb, **useCallbacks['CosAnnealMom']}))

    if 'ValidationMonitor' in useCallbacks:
        callbacks.append(ValidationMonitor(**useCallbacks['ValidationMonitor']))

    history = model.fit(**{**train, 'class_weight':'auto', #Training data
                         'validation_data':(val['x'], val['y'], val['sample_weight']), #Validation data 
                         'callbacks':callbacks, #Callbacks
                         **trainParams})
    if useEarlyStop:
        model.load_weights("train_weights/best.h5") #Load best model
    
    return model, history.history, callbacks

def foldPrep(training, validation, features, preprocParams):
    '''Prepare inputs, targets, and weights for training'''
    #Preprocess
    inputPipe = getPreProcPipes(**preprocParams)
    inputPipe.fit(training[features])
    
    #Inputs and targets
    train_X = inputPipe.transform(training[features].astype('float32'))
    val_X = inputPipe.transform(validation[features].astype('float32'))
    train_y = training['gen_target'].astype('int')
    val_y = validation['gen_target'].astype('int')
    
    #Sample weights - norm to signal + bkg = 2
    training.loc[training.gen_target == 0, 'gen_norm_weight'] = training.loc[training.gen_target == 0, 'gen_weight']/np.sum(training.loc[training.gen_target == 0, 'gen_weight'])
    training.loc[training.gen_target == 1, 'gen_norm_weight'] = training.loc[training.gen_target == 1, 'gen_weight']/np.sum(training.loc[training.gen_target == 1, 'gen_weight'])
    train_w = training['gen_norm_weight'].astype('float32')
    validation.loc[validation.gen_target == 0, 'gen_norm_weight'] = validation.loc[validation.gen_target == 0, 'gen_weight']/np.sum(validation.loc[validation.gen_target == 0, 'gen_weight'])
    validation.loc[validation.gen_target == 1, 'gen_norm_weight'] = validation.loc[validation.gen_target == 1, 'gen_weight']/np.sum(validation.loc[validation.gen_target == 1, 'gen_weight'])
    val_w = validation['gen_norm_weight'].astype('float32')
    val_w *= len(val_w)/len(train_w) #Re-norm val weights to account for differences in sample sizes

    return {'x':train_X, 'y':train_y, 'sample_weight':train_w}, {'x':val_X, 'y':val_y, 'sample_weight':val_w}

def cvTrainClassifier(data, features, nFolds, preprocParams, modelParams, trainParams, patience=10, useEarlyStop=True, useCallbacks={}, plot=True):
    '''Run model in CV over data and return results and mean performance'''
    start = timeit.default_timer()

    #Define holders for performance ledgers
    results = []
    histories = []

    #Reset temporary storage
    os.system("mkdir train_weights")
    os.system("rm train_weights/*.h5")
    os.system("rm train_weights/*.json")
    os.system("rm train_weights/*.pkl")

    #Initialise stratified k-fold splitter
    skf = StratifiedKFold(n_splits=nFolds, shuffle=True)

    for i, (trainIndeces, valIndeces) in enumerate(skf.split(data, data['gen_target'])): #test and train are sets of indices for the current CV fold
        fold = timeit.default_timer()
        print("Running fold", i+1, "/", nFolds)
        
        training = data.iloc[trainIndeces]
        validation = data.iloc[valIndeces]
        
        train, val = foldPrep(training, validation, features, preprocParams)

        #Train model as normal
        model = getModel(**modelParams)
        model, history, _ = trainClassifier(model, train, val, trainParams, patience=patience, useEarlyStop=useEarlyStop, saveBest=True, useCallbacks=useCallbacks)
        model.load_weights("train_weights/best.h5")
        
        #Compute and record performance and training history
        results.append({})
        validation['pred_class'] = model.predict(val['x'], verbose=0)
        results[-1]['loss'] = model.evaluate(verbose=0, **val) #Gets loss on validation data
        results[-1]['AUC'] = roc_auc_score(val['y'], validation['pred_class'].astype('float32'), sample_weight=val['sample_weight']) #Gets ROC AUC for validation data
        results[-1]['AMS'], results[-1]['cut'] = amsScanQuick(validation, wFactor=len(data)/len(validation))
        histories.append(history)                               

        print("Score is:", results[-1])
        print("Fold took {:.3f}s\n".format(timeit.default_timer() - fold))

    #Summarise training and results
    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if plot: getHistoryPlot(histories)
    for score in results[0]:
        mean = uncertRound(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
    
    return results, histories

def runLRFinder(data, features, modelParams, trainParams, preprocParams, 
                useValidation=False, lrBounds=[1e-7, 10], verbose=0, plot=True, nEpochs=1):
    '''Run LR finder callback over data for model and plot loss as function of learning rate'''
    start = timeit.default_timer()
    
    #Data prep
    trainIndeces, valIndeces = train_test_split([i for i in data.index.tolist()], test_size=0.2)
    train, val = foldPrep(data.iloc[trainIndeces], data.iloc[valIndeces], features, preprocParams)
    
    params = {'nSteps':math.ceil(nEpochs*len(trainIndeces)/trainParams['batch_size']),
              'lrBounds':lrBounds, 'verbose':verbose}
    if useValidation:
        params = {'valData':val, 'valBatchSize':trainParams['batch_size'], **params}
    lrFinder = LRFinder(**params)
    
    model = getModel(**modelParams)
    model.fit(**{**train, 'class_weight':'auto',
                 'callbacks':[lrFinder],
                 'batch_size':trainParams['batch_size'],
                 'epochs':nEpochs, 'verbose':1})

    print("\n______________________________________")
    print("Training finished")
    print("LR finder took {:.3f}s ".format(timeit.default_timer() - start))
    if plot:
        lrFinder.plot_lr()    
        lrFinder.plot(n_skip=10)
        if useValidation:
            lrFinder.plot_genError()
    print("______________________________________\n")
        
    return lrFinder

def timeBatchsize(modelParams, train, val, lr, batchsize, trainParams):
    '''Get the per epoch train-time for a given batch size'''
    model = getModel(**{'version':modelParams['version'], 'nIn':modelParams['nIn'],
                        'compileArgs':{**modelParams['compileArgs'], 'lr':lr}}) 
    start = timeit.default_timer()
    history = model.fit(**{**train, 'class_weight':'auto',
                           'validation_data':(val['x'], val['y'], val['sample_weight']),
                           'batch_size':batchsize,
                           **trainParams})
    time = (timeit.default_timer()-start)/trainParams['epochs']
    
    minEpochs = np.argmin(history.history['val_loss'])
    print("Minimum of {:.8f} achieved in {} epochs, {:.2f} seconds/epoch".format(history.history['val_loss'][minEpochs],
                                                                                 minEpochs, time))
    return history, time