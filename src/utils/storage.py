from pathlib import Path

import pandas as pd


def update_csv(df, file_path, key_columns):
    """Обновляет локальный CSV и возвращает полный набор данных."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        old_df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )

        old_df = old_df.drop_duplicates(
            subset=key_columns,
            keep="last",
        )

        old_rows = len(old_df)

        combined = pd.concat(
            [old_df, df],
            ignore_index=True,
        )
    else:
        old_rows = 0
        combined = df.copy()

    combined = combined.drop_duplicates(
        subset=key_columns,
        keep="last",
    )

    new_rows = max(
        len(combined) - old_rows,
        0,
    )

    combined.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig",
    )

    return combined, new_rows