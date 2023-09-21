import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from Helper_Util import Baseline_score, Baseline_predict
from Feature_Engineering_Util import baseline_preprocessing, preprocessing, Load_Data

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
cbc = CatBoostRegressor(loss_function='RMSE',custom_metric=['RMSE','MAE','Huber:delta=0.5'],learning_rate=0.043)
lgb = LGBMRegressor(objective='rmse',metric=['l1', 'l2', 'huber'])
xgb = XGBRegressor(objective='reg:squarederror',eval_metric=['mae', 'rmse', 'mape'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loading")
    parser.add_argument(
        "--dataset_path",
        help="Path to Datasets CSV",
        default=
        'datasets'
    )
    parser.add_argument(
        "--verbose_value",
        help="Number of iterations for printout",
        default=0
    )
    parser.add_argument(
        "--split_method",
        help="Type of Split Method, KFold or Train_Test_Split",
        default="KFold"
    )
    parser.add_argument(
        "--num_splits",
        help="Using KFOLD, Set Number of Splits",
        default=5
    )
    parser.add_argument(
        "--random_seed",
        help="RANDOM SEED",
        default=0
    )
    args = parser.parse_args()

    print("=="*10,"LOADING DATA","=="*10)
    Train, Test, Sub = Load_Data(args.dataset_path)
    print("=="*10,"BASELINE PREPROCESSING DATA","=="*10)
    TT, tests = baseline_preprocessing(Train, Test)
    Baseline = Baseline_score(train=TT,
                              model=cbc,
                              split_method=args.split_method,
                              num_split=int(args.num_splits),
                              VERBOSE=int(args.verbose_value),
                              random_seed=int(args.random_seed))
    print("=="*10,"GETTING A BASELINE SCORE","=="*10)
    lgb_score,bsv_1 = Baseline.run()
    print(f"CV_RMSE_Score:{lgb_score}")

    print(f"Before F.E No of Columns: {Train.shape[1]}")
    print("=="*10,"FEATURE ENGINEERING","=="*10)
    TT, tests = preprocessing(Train, Test)
    print(f"After F.E No of Columns: {TT.shape[1]}")

    Baseline = Baseline_score(train=TT,
                              model=cbc,
                              split_method=args.split_method,
                              num_split=int(args.num_splits),
                              VERBOSE=int(args.verbose_value),
                              random_seed=int(args.random_seed))
    print("=="*10,"GETTING A SCORE","=="*10)
    lgb_score,bsv_1 = Baseline.run()
    print(f"CV_RMSE_Score:{lgb_score}")

    getpreds = Baseline_predict(train=TT,
                                   model=lgb,
                                   split_method=args.split_method,
                                   num_split=int(args.num_splits),
                                   return_baseline=True,
                                   VERBOSE=int(args.verbose_value),
                                   random_seed=int(args.random_seed))
    print("=="*10,"GETTING PREDICTIONS","=="*10)
    predict,bsv_test = getpreds.predict(tests)
    predictions_df = pd.DataFrame({'ID': Test.ID, 'price': sum(predict)/int(args.num_splits)})
    predictions_df.to_csv(f"{args.dataset_path}/predictions.csv",index=False)
    print("=="*10,"DONE","=="*10)
  