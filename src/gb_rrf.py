import pickle

import matplotlib.pyplot as plt
import numpy as np

from data_utils import DataUtils as du

def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

def get_predicions(model_reg,model_cas,model_name):
    df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        '../data/train.csv', '../data/test.csv', output_cols=['registered', 'casual', 'count'],model = model_name,normalize=False)

    y_pred_val_reg = np.exp(model_reg.predict(val_x)) - 1
    y_pred_val_cas = np.exp(model_cas.predict(val_x)) - 1
    y_pred_train_reg = np.exp(model_reg.predict(train_x)) - 1
    y_pred_train_cas = np.exp(model_cas.predict(train_x)) - 1
    y_pred_test_reg = np.exp(model_reg.predict(test_x)) - 1
    y_pred_test_cas = np.exp(model_cas.predict(test_x)) - 1

    y_pred_val = np.round(y_pred_val_reg + y_pred_val_cas)
    y_pred_val[y_pred_val < 0] = 0

    y_pred_train= np.round(y_pred_train_reg + y_pred_train_cas)
    y_pred_train[y_pred_train < 0] = 0

    y_pred_test= np.round(y_pred_test_reg + y_pred_test_cas)
    y_pred_test[y_pred_test < 0] = 0

    return y_pred_val,y_pred_train,y_pred_test,val_y,train_y['count'],test_date_df

if __name__ == '__main__':

    rrf_reg_model = pickle.load(open('rrf_reg_model.sav', 'rb'))
    rrf_cas_model = pickle.load(open('rrf_cas_model.sav', 'rb'))
    gbm_reg_model = pickle.load(open('gbm_reg_model.sav', 'rb'))
    gbm_cas_model = pickle.load(open('gbm_cas_model.sav', 'rb'))

    rrf_pred_val,rrf_pred_train,rrf_pred_test,val_y,train_y,test_date_df = get_predicions(rrf_reg_model,rrf_cas_model,'rrf')
    gb_pred_val,gb_pred_train,gb_pred_test,_,_,_ = get_predicions(gbm_reg_model,gbm_cas_model,'gb')

    alfa_params = np.arange(0.8,0.81,0.05)

    val_scores = np.zeros(alfa_params.shape[0])
    train_scores = np.zeros(alfa_params.shape[0])

    for i,alfa in enumerate(alfa_params):
        blend_pred_val = alfa*gb_pred_val + (1-alfa)*rrf_pred_val
        val_score = get_rmsle(blend_pred_val, val_y)
        val_scores[i] = val_score

        blend_pred_train = alfa*gb_pred_train + (1-alfa)*rrf_pred_train
        train_score = get_rmsle(blend_pred_train, train_y)
        train_scores[i] = train_score

        blend_pred_test = alfa * gb_pred_test + (1 - alfa) * rrf_pred_test

        print("Val error:",val_score,", Train score:",train_score)

    #Submiting predictions
    test_date_df['count'] = blend_pred_test
    test_date_df.to_csv('predictions_gmb_rrf.csv', index=False)

    plt.plot(alfa_params, val_scores)
    #plt.plot(alfa_params, train_scores)
    plt.ylabel('RMSLE', fontsize=18)
    plt.xlabel('alfa', fontsize=18)
    plt.legend(['validation error', 'train error'], fontsize=14)
    plt.show()