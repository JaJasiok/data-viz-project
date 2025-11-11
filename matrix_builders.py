import pandas as pd

from data_processing import ordered_rows


# ---------- MATRIX BUILDERS ----------

def money_in_matrix(top_club_ids, transfers_enriched):
    """
    Spending per club on incoming players for each country.
    Rows: origin country (from_country)
    Cols: buying club (to_club_id)
    """
    tf = transfers_enriched[
        transfers_enriched['to_club_id'].isin(top_club_ids)
    ].copy()
    tf = tf[tf['fee_eur'] > 0]
    tf = tf[~tf['from_country'].isin(['Without Club', 'Retired'])]

    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['from_country', 'to_club_id'])['fee_eur']
          .sum()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index))
    return mat

def money_out_matrix(top_club_ids, transfers_enriched):
    """
    Incomings (money) per club from outgoing players for each country.
    Rows: destination country (to_country)
    Cols: selling club (from_club_id)
    """
    tf = transfers_enriched[
        transfers_enriched['from_club_id'].isin(top_club_ids)
    ].copy()
    tf = tf[tf['fee_eur'] > 0]
    tf = tf[~tf['to_country'].isin(['Without Club', 'Retired'])]

    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['to_country', 'from_club_id'])['fee_eur']
          .sum()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index))
    return mat

def players_in_matrix(top_club_ids, transfers_enriched):
    """
    Number of players received from each country per club.
    Rows: from_country (+ 'Without Club')
    Cols: to_club_id
    """
    tf = transfers_enriched[
        transfers_enriched['to_club_id'].isin(top_club_ids)
    ].copy()
    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['from_country', 'to_club_id'])['player_id']
          .count()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index, extras=('Without Club',)))
    return mat

def players_out_matrix(top_club_ids, transfers_enriched):
    """
    Number of players sent to each country per club.
    Rows: to_country, with extra:
      - 'Without Club' for leaving without new club
      - 'Retired' for retirements
    Cols: from_club_id
    """
    tf = transfers_enriched[
        transfers_enriched['from_club_id'].isin(top_club_ids)
    ].copy()
    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['to_country', 'from_club_id'])['player_id']
          .count()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(
        index=ordered_rows(mat.index, extras=('Without Club', 'Retired'))
    )
    return mat

def column_percentage_matrix(matrix):
    """
    For each club (column), convert counts to percentages of that club's total.
    Columns summing to 0 become all 0.
    """
    if matrix is None or matrix.empty:
        return matrix

    pct = matrix.astype(float).copy()
    col_sums = pct.sum(axis=0)

    for col in pct.columns:
        total = col_sums[col]
        if total > 0:
            pct[col] = pct[col] / total * 100.0
        else:
            pct[col] = 0.0

    return pct