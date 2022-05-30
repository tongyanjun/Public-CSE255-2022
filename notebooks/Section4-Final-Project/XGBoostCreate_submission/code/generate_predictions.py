for fold_i in range(len(folds)):
    fold=folds[fold_i]

    #load table entries
    test_csv=f'../public_tables/{fold["in"]}'

    test=pd.read_csv(test_csv,index_col=0)
    test.index=test['filename']
    test.shape

    out=pd.DataFrame()
    out.index=test.index
    out['label']=0
    out['urban']=test['urban']

    ## Encode all data using encoding tree
    Enc_data=encoded_dataset(image_dir,out,tree,label_col='label')

    data=to_DMatrix(Enc_data.data)
    Preds=zeros([Enc_data.data.shape[0],len(bst_list)])
    for i in range(len(bst_list)):
        Preds[:,i]=bst_list[i].predict(data,output_margin=True)
    Preds=(Preds-scaling_mean)/scaling_std # apply overall score scaling

    _mean=np.mean(Preds,axis=1)
    _std=np.std(Preds,axis=1)

    pred_wo_abstention=(2*(_mean>0))-1
    pred_with_abstention=copy(pred_wo_abstention)
    pred_with_abstention[_std>abs(_mean)]=0

    out['pred_with_abstention'] = pred_with_abstention
    out['pred_wo_abstention'] = pred_wo_abstention


    outFile=f'data/{fold["out"]}'
    out.to_csv(outFile)
    print('\n\n'+'-'*60)
    print(outFile)
