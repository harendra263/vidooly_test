import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import re
from typing import List
from tqdm import tqdm


def get_time(time_part: str, duration: List[str], time_part_name: str, df: pd.DataFrame) -> pd.DataFrame:
    for idx, time in enumerate(duration):
        rgx = re.search(pattern=f"\d+{time_part}", string=time)
        df.loc[idx, time_part_name] = f"0{time_part}" if rgx is None else rgx.group()
    df[time_part_name] = df[time_part_name].str.replace(time_part, "")
    df[time_part_name] = df[time_part_name].astype(int)
    return df


def convert_to_numeric(colname: List[str], df: pd.DataFrame) ->pd.DataFrame:
    for ele in tqdm(range(len(colname))):
        if df[df[colname[ele]]=='F'].empty:
            df = df
        else:
            df = df.drop(df[df[col_to_numeric[ele]]=='F'].index, axis=0)
    df[colname] = df[colname].astype(int)
    return df

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")

    # One of the columns is video duration and it is in string. Need to convert the string into time

    pattern= '\d+\w+'
    df['video_duration'] = df.duration.apply(lambda x: re.findall(pattern=pattern, string=x)[0])

    df = get_time(time_part='H', duration=df.video_duration.values, time_part_name='Hour', df=df)
    df = get_time(time_part='M', duration=df.video_duration.values, time_part_name='Minute', df=df)
    df = get_time(time_part='S', duration=df.video_duration.values, time_part_name='Second', df=df)

    df["Second_to_min"] = df["Second"] / 60
    df["Hour_to_min"] = df["Hour"] *60
    df["Total_Time"] = df["Second_to_min"] + df["Hour_to_min"] + df["Minute"]
    
    label_encoder = LabelEncoder()
    df['category_enc'] = label_encoder.fit_transform(df.category)

    df.drop(["vidid", "category", "duration", "video_duration", "published", "Hour", 
            "Minute", "Second", "Second_to_min", "Hour_to_min"], axis=1, inplace= True)
    
    col_to_numeric = df.columns[df.dtypes=="object"]

    df = convert_to_numeric(colname=col_to_numeric, df=df)


    # Standardizating the numeric vars
    # scaler = MinMaxScaler()
    # col_names = df.columns
    # d = scaler.fit_transform(df)
    # scaled_df = pd.DataFrame(d, columns=col_names)

    df.to_csv("input/scaled_df.csv", index=False)


    