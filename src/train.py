import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=120):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

df = pd.read_csv("input/train.csv")


df['video_duration'] = df.duration.apply(lambda x: re.findall(pattern='\d+\w+', string=x)[0])

def get_time(time_ext: str, 
             duration: List[str], 
             time_name: str, 
             df: pd.DataFrame
             ) -> pd.DataFrame:
    for idx, x in enumerate(duration):
        m = re.search(pattern=f"\d+{time_ext}", string=x)
        df.loc[idx, time_name] = m.group() if m is not None else f'0{time_ext}'
    df[time_name] = df[time_name].str.replace(time_ext, '')
    df[time_name] = df[time_name].astype(int)
    return df

df = get_time(time_ext='H', duration=df.video_duration.values, time_name='Hour', df=df)
df = get_time(time_ext='M', duration=df.video_duration.values, time_name='Minute', df=df)
df = get_time(time_ext='S', duration=df.video_duration.values, time_name='Second', df=df)

df["Second_to_min"] = df["Second"] / 60
df["Hour_to_min"] = df["Hour"] *60
df["Total_Time"] = df["Second_to_min"] + df["Hour_to_min"] + df["Minute"]
df.drop(["Hour", "Minute", "Second", "Second_to_min", "Hour_to_min"], axis=1, inplace= True)

label_enc = LabelEncoder()
df['category_enc'] = label_enc.fit_transform(df.category)

col_to_numeric = ['adview', 'views', 'likes', 'dislikes', 'comment', 'Total_Time', 'category_enc']

for ele in tqdm(range(len(col_to_numeric))):
    if df[df[col_to_numeric[ele]]=='F'].empty:
        df= df
    else:
        df = df.drop(df[df[col_to_numeric[ele]]=='F'].index, axis=0)

df[col_to_numeric] = df[col_to_numeric].astype(int)

# =============================================
#  PLOT RELATION BETWEEN FEATURES
# =============================================

fig, axes= plt.subplots(nrows=3, ncols=2, figsize=(15, 8))
fig.suptitle("No of Adviews")

axes[0, 0].scatter(df.views.values,df.comment.values)
axes[0, 0].set_title("Views vs Comment")
axes[0, 0].set_xlabel("Views")
axes[0, 0].set_ylabel("Comments")

axes[0, 1].scatter(df.views.values,df.Total_Time.values, color="tab:orange")
axes[0, 1].set_title("Views vs Total Time")
axes[0, 1].set_xlabel("Views")
axes[0, 1].set_ylabel("Total Time")

axes[1, 0].scatter(df.views.values,df.dislikes.values, color="tab:green")
axes[1, 0].set_title("Views vs Dislikes")
axes[1, 0].set_xlabel("Views")
axes[1, 0].set_ylabel("Dislikes")



axes[1, 1].scatter(df.likes.values,df.views.values, marker='o', color="tab:red")
axes[1, 1].set_title("Views vs Likes")
axes[1, 1].set_xlabel("Views")
axes[1, 1].set_ylabel("Likes")

axes[2, 0].scatter(df.views.values,df.adview.values, color="tab:pink")
axes[2, 0].set_title("Views vs Adview")
axes[2, 0].set_xlabel("Views")
axes[2, 0].set_ylabel("Adview")

axes[2, 1].bar(df.category.values,df.adview.values, color="tab:grey")
axes[2, 1].set_title("Category vs Adview")
axes[2, 1].set_xlabel("Category")
axes[2, 1].set_ylabel("Adview")

plt.tight_layout()
save_fig("Views_vs_All_Features")

# =====================================================================

cols_to_drop = ['vidid', 'published', 'duration','category', 'video_duration']
df.drop(cols_to_drop, axis=1, inplace=True)

# ===============================================================
#  PLOT CORRELATION PLOT
# ===============================================================

correlations = df.corr()
# annot=True displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heatmap of YouTube Data - Pearson Correlations');
save_fig("Correlation_plot")

# =============================================================
#  MODEL BUILDING
# =======================================================

from sklearn.model_selection import train_test_split

SEED = 42

X = df.drop("adview", axis=1)
y = df["adview"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

regr = RandomForestRegressor(max_depth=2, random_state=SEED)
regr.fit(X_train, y_train)

# Train Score
train_score = regr.score(X_train, y_train)

# Test Score
test_score = regr.score(X_test, y_test)

# Write scores to a file
with open('input/metrics.txt', 'w') as outfile:
    outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    outfile.write("Test varinace explained: %2.1f%%\n" % test_score)

# ============================================
#  FEATURE IMPORTANCE
# ============================================

importance = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importance)), columns=['Features', 'Importances'])


# Image formatting

axis_fs = 18
title_fs = 22
sns.set(style="whitegrid")

ax = sns.barplot(x = "Features", y="Importances", data=feature_df)
ax.set_xlabel("Feature", fontsize=axis_fs)
ax.set_ylabel("Importance", fontsize = axis_fs)

ax.set_title("Random Forest\nFeature Importance", fontsize= title_fs)

plt.tight_layout()
plt.savefig("images/end_to_end_project/Feature_Importance.png", dpi=120)
plt.close()

# ==================================================
# PLOT RESIDUALS
# ==================================================

y_pred = regr.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True Adview',fontsize = axis_fs) 
ax.set_ylabel('Predicted Adview', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, "residuals.png"),dpi=120)

