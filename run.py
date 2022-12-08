import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from models import RandomModel, MostPopular, ModelALS, BIVAE, evaluate
from setup import ITEM_COL, GENRE_COL, TOP_K_METRICS, TOP_K_PRED


def main(out_folder='outputs'):
    df = pd.read_csv('data/personalize.zip', compression='zip').iloc[:, :3]

    genres = pd.read_csv('genres.csv').rename({"movieId": ITEM_COL}, axis=1).dropna()

    train, test = train_test_split(df, test_size=None, train_size=0.75, random_state=42)

    metrics = {}
    for model_cls in [BIVAE, ModelALS, RandomModel, MostPopular]:
        model = model_cls()
        model.fit(train)

        preds = model.transform(TOP_K_PRED)

        preds = preds.merge(genres[[ITEM_COL, GENRE_COL]], on=ITEM_COL, how='left')
        preds.to_csv(Path(out_folder) / f"{model_cls.__name__}_preds.csv", index=False)
        metrics[model_cls.__name__] = evaluate(train, test, preds, TOP_K_METRICS)

    with open('outputs/metrics.json', 'w') as fp:
        json.dump(metrics, fp)


if __name__ == "__main__":
    main()
