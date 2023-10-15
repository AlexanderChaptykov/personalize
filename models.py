import cornac
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import Window, SparkSession
from pyspark.sql.functions import rank, col
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation
from recommenders.models.cornac.cornac_utils import predict_ranking

from setup import USER_COL, ITEM_COL, RATING_COL, PREDICTION_COL, GENRE_COL

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
# comment

class AbstractModel:
    user_col = USER_COL
    item_col = ITEM_COL
    rating_col = RATING_COL
    score_col = PREDICTION_COL
    train = None

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError


class ModelALS(AbstractModel):
    def __init__(self):
        self.model = ALS(
            rank=10,
            maxIter=15,
            implicitPrefs=False,
            regParam=0.05,
            coldStartStrategy="drop",
            nonnegative=False,
            seed=42,
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
        )

    def fit(self, train):
        self.train = spark.createDataFrame(train)
        self.model = self.model.fit(self.train)

    def transform(self, top_k_pred):
        users = self.train.select(self.user_col).distinct()
        items = self.train.select(self.item_col).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = self.model.transform(user_item)

        # Remove seen items.
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            self.train.alias("train"),
            (dfs_pred[self.user_col] == self.train[self.user_col])
            & (dfs_pred[self.item_col] == self.train[self.item_col]),
            how="outer",
        )

        top_all = dfs_pred_exclude_train.filter(
            dfs_pred_exclude_train[f"train.{self.rating_col}"].isNull()
        ).select(
            "pred." + self.user_col, "pred." + self.item_col, "pred." + self.score_col
        )

        window = Window.partitionBy(top_all[self.user_col]).orderBy(
            top_all[self.score_col].desc()
        )

        # Getting results
        # noinspection PyTypeChecker
        preds = (
            top_all.select("*", rank().over(window).alias("rank"))
            .filter(col("rank") <= top_k_pred)
            .drop("rank")
            .toPandas()
        )
        return preds


class BIVAE(AbstractModel):
    def __init__(
        self,
        latent_dim=50,
        act_fn="tanh",
        likelihood="pois",
        n_epochs=50,
        batch_size=128,
        lr=0.001,
        random_seed=42,
    ):
        self.model = cornac.models.BiVAECF(
            k=latent_dim,
            encoder_structure=[100],
            act_fn=act_fn,
            likelihood=likelihood,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=random_seed,
            verbose=False,
        )

    def fit(self, train, seed=42):
        self.train = train
        self.model = self.model.fit(
            cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=seed)
        )

    def transform(self, top_k_pred):
        preds = predict_ranking(
            self.model,
            self.train,
            usercol=self.user_col,
            itemcol=self.item_col,
            remove_seen=True,
        )
        preds = (
            preds.sort_values([self.user_col, self.score_col], ascending=False)
            .groupby(self.user_col)
            .head(top_k_pred)
        )
        return preds


class MostPopular(AbstractModel):
    def fit(self, train, seed=42):
        self.model = (
            train[self.item_col].value_counts()
            / train[self.item_col].value_counts().max()
        )
        self.train = train

    def transform(self, top_k_pred):
        dfs = []
        for i in self.train[self.user_col].unique():
            visited = set(
                self.train[self.train[self.user_col] == i][self.item_col].unique()
            )
            predicted = self.model[~self.model.index.isin(visited)][:top_k_pred]

            dfs.append(
                pd.DataFrame(
                    {
                        self.user_col: i,
                        self.item_col: predicted.index,
                        self.score_col: predicted.values,
                    }
                )
            )
        return pd.concat(dfs)


class RandomModel(AbstractModel):
    uniques = None

    def fit(self, train, seed=42):
        self.train = train
        self.all_items = set(train[self.item_col].unique())

    def transform(self, top_k_pred):
        dfs = []
        for i in self.train[self.user_col].unique():
            visited = set(
                self.train[self.train[self.user_col] == i][self.item_col].unique()
            )
            sample_items = self.all_items - visited
            predicted = pd.Series(list(sample_items)).sample(top_k_pred).values

            dfs.append(
                pd.DataFrame(
                    {self.user_col: i, self.item_col: predicted, self.score_col: 1}
                )
            )
        return pd.concat(dfs)


def _genre_maper(other_genres):
    def maper(x: str, other_genres=set(other_genres)):
        if type(x) == str:
            x = x.split("|")[0]
            if x in other_genres:
                return "Other"
            else:
                return x
        return

    return maper


def _genre_maper2():
    def maper(x: str):
        if type(x) == str:
            x = x + "_" if "|" not in x else x
            x = x.split("|")[0]
        return x

    return maper


def evaluate(
    train,
    test,
    preds,
    top_k,
    user_col=USER_COL,
    item_col=ITEM_COL,
    rating_col=RATING_COL,
    pred_col=PREDICTION_COL,
    divers_col=GENRE_COL,
):
    diversity = (
        preds.dropna()
        .assign(genre=preds[divers_col].map(_genre_maper2()))
        .groupby(user_col)["genre"]
        .nunique()
        .median()
    )
    coverage = preds[item_col].nunique() / train[item_col].nunique()
    preds = (
        preds.sort_values([user_col, pred_col], ascending=False)
        .groupby(user_col)
        .head(top_k)
    )

    spark_metrics = SparkRankingEvaluation(
        spark.createDataFrame(test),
        spark.createDataFrame(preds.drop(divers_col, axis=1)),
        k=top_k,
        col_user=user_col,
        col_item=item_col,
        col_rating=rating_col,
        col_prediction=pred_col,
        relevancy_method="top_k",
    )
    return {
        "Precision@k": spark_metrics.precision_at_k(),
        "MAP@k": spark_metrics.map_at_k(),
        "Coverage": coverage,
        "Diversity": diversity,
    }
