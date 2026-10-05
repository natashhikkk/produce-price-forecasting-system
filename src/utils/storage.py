from pathlib import Path

import pandas as pd


def update_csv(df, file_path, key_columns):
    """
    Сохраняет данные в CSV и при повторном запуске
    не создаёт дубликаты.

    Возвращает количество новых записей.
    """

    file_path = Path(file_path)

    # Создаём папку, если её ещё нет
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Если файл ещё не существует,
    # сохраняем все полученные данные
    if not file_path.exists():

        df.to_csv(
            file_path,
            index=False,
            encoding="utf-8-sig"
        )

        return len(df)

    # Загружаем ранее сохранённые данные
    old_df = pd.read_csv(file_path)

    # Объединяем старые и новые данные
    combined = pd.concat(
        [old_df, df],
        ignore_index=True
    )

    # Количество строк до удаления дублей
    rows_before = len(combined)

    # Удаляем повторные наблюдения
    combined = combined.drop_duplicates(
        subset=key_columns,
        keep="last"
    )

    new_rows = len(combined) - len(old_df)

    # Сохраняем обновлённый файл
    combined.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    return max(new_rows, 0)